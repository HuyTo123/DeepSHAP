import warnings

import numpy as np
from packaging import version

from .deep_utils import _check_additivity


class PyTorchDeep():
    def __init__(self, model, data):
        import torch
        print("PyTorchDeep.__init__")
        if version.parse(torch.__version__) < version.parse("0.4"):
            warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]
        self.data = data # data is a list of tensors - baseline data
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        self.count  = 0
        # interim use for you want to get the output of a specific layer in the model
        # Tuple at this point is (model, layer) where model is the model and layer is the layer you want to get the output of
        if isinstance(model, tuple):
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
    

        self.model = model.eval()
        # Assume that the model is only one output
        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            #  * = unpacking argument 
            outputs = model(*data) # Models forward pass, shape of output is (batch_size, num_outputs)

            # also get the device everything is running on
            self.device = outputs.device
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu().numpy()
    # for interim layer
    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        # forward_handle = add_interim_values
        # backward_handle = deeplift_grad
        handles_list = []
        #  Take all layer of the model
        model_children = list(model.children())
        if model_children:
            for child in model_children:
                handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
              
        else:  # leaves
            #  In pytorch, the leaves are the layers that are not containers, and we only need to supply the hooks which is action we onlyneed
            #  x and y and all values related will automatic calculate by pytorch when forward and backward pass
            # actually it just registers the event handlers at layer which fitted the rules
            handles_list.append(model.register_forward_hook(forward_handle))
            handles_list.append(model.register_full_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if "nn.modules.container" in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    def gradient(self, idx, inputs):
        import torch
        #  remove all gradient parameters to make sure 
        #  we don't use the old ones
        self.model.zero_grad()
        # inputs is joint_x (type array), idx is class of sample J 
        X = [x.requires_grad_() for x in inputs]
        # forward pass
        outputs = self.model(*X)
        # outputs is tensor [batch_size, num_outputs] 
        selected = [val for val in outputs[:, idx]]
        print(selected[0])
        grads = []
        if self.interim:
            interim_inputs = self.layer.target_input
            for idx, input in enumerate(interim_inputs):
                grad = torch.autograd.grad(
                    selected, input, retain_graph=True if idx + 1 < len(interim_inputs) else None, allow_unused=True
                )[0]
                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = torch.zeros_like(X[idx]).cpu().numpy()
                grads.append(grad)
            del self.layer.target_input
            return grads, [i.detach().cpu().numpy() for i in interim_inputs]
        else:
            for idx, x in enumerate(X):
                grad = torch.autograd.grad(
                    selected, x, retain_graph=True if idx + 1 < len(X) else None, allow_unused=True
                )[0]
                if grad is not None:
                    grad = grad.cpu().numpy()
                else:
                    grad = torch.zeros_like(x).cpu().numpy()
                grads.append(grad)
            return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max", check_additivity=True):
        import torch
      
        # check if we have multiple inputs, if not, we need to convert the input to a list
        if not self.multi_input:
            # We recheck the type of X, because it may be a single tensor or a list of tensors
            assert not isinstance(X, list), "Expected a single tensor model input!"
            X = [X]
        else:
            # if multi_input, we need to check that is list of tensors
            assert isinstance(X, list), "Expected a list of model inputs!"

        # X is model inputs, so we need to detach them from the computing graph, to avoid any gradient computation
        # and move them to the same device as the model
        # If we use detach(), 2 tensor will not independent, if one change the other will change too
        X = [x.detach().to(self.device) for x in X]

        model_output_values = None
       
        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain, _ is the values, model_output_ranks are the ranks indexes
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            # max_abs is the absolute value of the model output, so we sort by the absolute value of the model output
            elif output_rank_order == "max_abs":  
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                emsg = "output_rank_order must be max, min, or max_abs!"
                raise ValueError(emsg)
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            #  User did not specify ranked_outputs, so we will use all outputs without ranking
            # tensor(batch_size, num_outputs) = torch.ones((X[0].shape[0], self.num_outputs)).int() * torch.arange(0, self.num_outputs).int()
            # element wise
            model_output_ranks = (
                torch.ones((X[0].shape[0], self.num_outputs)).int() * torch.arange(0, self.num_outputs).int()
            )
        # Register the hook for forward and backward pass
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)
        # add the specialized gradient handles for the interim layer
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(np.zeros((X[0].shape[0],) + self.interim_inputs_shape[k][1:]))
            else:
                # assign the input tensor of a picture 
                for k in range(len(X)):
                    phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                # tile (Nhân bản) the inputs to line up with (Thẳng hàng, tương ứng) the background data samples
                # data is baseline
                # we use j : j +1 to keep the dimension 1, [1, 3, 32, 32] instead of [3, 32, 32]
                # t always is 0 if we don't have multi_input
                tiled_X = [
                    # X[t][j : j + 1].repeat((self.data[t].shape[0],)  = take input of tensor batch_size 1 and repeat
                    # we make tuple (batch_size baseline, 1,1,1,1) to make it same shape with baseline data
                    X[t][j : j + 1].repeat((self.data[t].shape[0],) + tuple([1 for k in range(len(X[t].shape) - 1)]))
                    for t in range(len(X))
                ]
                # concat the input and background data [batch_size X + batch_size baseline, 3, 32, 32]
                joint_x = [torch.cat((tiled_X[t], self.data[t]), dim=0) for t in range(len(X))]
                # run attribution computation graph
                # take the class i of j batch_size , tensor rank 0 scalar
                feature_ind = model_output_ranks[j, i]

                # Error in here !! call only one
                sample_phis = self.gradient(feature_ind, joint_x)

                
                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for k in range(len(output)):
                        x_temp, data_temp = np.split(output[k], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    for t in range(len(self.interim_inputs_shape)):
                        phis[t][j] = (sample_phis[t][self.data[t].shape[0] :] * (x[t] - data[t])).mean(0)
                else:
                    for t in range(len(X)):
                        phis[t][j] = (
                            (
                                torch.from_numpy(sample_phis[t][self.data[t].shape[0] :]).to(self.device)
                                * (X[t][j : j + 1] - self.data[t])
                            )
                            .cpu()
                            .detach()
                            .numpy()
                            .mean(0)
                        )
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        # check that the SHAP values sum up to the model output
        if check_additivity:
            if model_output_values is None:
                with torch.no_grad():
                    model_output_values = self.model(*X)

            _check_additivity(self, model_output_values.cpu(), output_phis)

        if isinstance(output_phis, list):
            # in this case we have multiple inputs and potentially multiple outputs
            if isinstance(output_phis[0], list):
                output_phis = [np.stack([phi[i] for phi in output_phis], axis=-1) for i in range(len(output_phis[0]))]
            # multiple outputs case
            else:
                output_phis = np.stack(output_phis, axis=-1)
        if ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis


# Module hooks


def deeplift_grad(module, grad_input, grad_output):
    """The backward hook which computes the deeplift
    gradient for an nn.Module
    """
    # first, get the module type
    module_type = module.__class__.__name__

    # first, check the module is supported
    if module_type in op_handler:
        # print(module_type,op_handler[module_type], op_handler[module_type].__name__) 
        # ex: Linear <function linear_1d at 0x000002AC8DE81580> linear_1d
        if op_handler[module_type].__name__ not in ["passthrough", "linear_1d"]:
            # op_handler[module_type] is a function, so we need to register hook to specific layer type
            return op_handler[module_type](module, grad_input, grad_output)
    else:
        warnings.warn(f"unrecognized nn.Module: {module_type}")
        return grad_input


def add_interim_values( module, input, output):
    """The forward hook used to save interim tensors, detached
    from the graph. Used to calculate the multipliers
    interim_values is temporarily saved in the module, and will be removed.
    Module.x and module.y will only use in backward pass
    """
    import torch
    # remove the x and y attributes if they exist, to avoid overwriting
    #  remove the x and y attributes if they exist, to avoid overwriting
    try:
        del module.x
    except AttributeError:
        pass
    try:
        del module.y
    except AttributeError:
        pass
    #  type module_type is string
    module_type = module.__class__.__name__
    #  Make sure the module name is defined in op_handler at below the class
    if module_type in op_handler:
        func_name = op_handler[module_type].__name__
        # First, check for cases where we don't need to save the x and y tensors
        if func_name == "passthrough":
            pass
        else:
            # check only the 0th input varies 
            # print(input[0].shape, output.shape, func_name) ~ input is tuple and output is tensor
            for i in range(len(input)):
                # output is a tensor so we need a little fix right here (tuple -> tensor)
                if i != 0 and type(output) is torch.Tensor:
                    assert input[i] == output[i], "Only the 0th input may vary!"
            # if a new method is added, it must be added here too. This ensures tensors
            # are only saved if necessary
            if func_name in ["maxpool", "nonlinear_1d"]:
                # only save tensors if necessary and will use at the backward pass
                if type(input) is tuple:
                    module.x = torch.nn.Parameter(input[0].detach())
                else:
                    module.x = torch.nn.Parameter(input.detach())
                if type(output) is tuple:
                    module.y = torch.nn.Parameter(output[0].detach())
                else:
                    module.y = torch.nn.Parameter(output.detach())



def get_target_input(module, input, output):
    """A forward hook which saves the tensor - attached to its graph.
    Used if we want to explain the interim outputs of a model
    """
    try:
        del module.target_input
    except AttributeError:
        pass
    module.target_input = input


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    import torch
    pool_to_unpool = {
        "MaxPool1d": torch.nn.functional.max_unpool1d,
        "MaxPool2d": torch.nn.functional.max_unpool2d,
        "MaxPool3d": torch.nn.functional.max_unpool3d,
    }
    pool_to_function = {
        "MaxPool1d": torch.nn.functional.max_pool1d,
        "MaxPool2d": torch.nn.functional.max_pool2d,
        "MaxPool3d": torch.nn.functional.max_pool3d,
    }
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x, module.kernel_size, module.stride, module.padding, module.dilation, module.ceil_mode, True
        )
        xmax_pos, rmax_pos = torch.chunk(
            pool_to_unpool[module.__class__.__name__](
                grad_output[0] * diffs, indices, module.kernel_size, module.stride, module.padding, list(module.x.shape)
            ),
            2,
        )

    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(
        torch.abs(delta_in) < 1e-7, torch.zeros_like(delta_in), (xmax_pos + rmax_pos) / delta_in
    ).repeat(dup0)

    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None




def nonlinear_1d(module, grad_input, grad_output):
    import torch

    # Tính delta_out: Khác biệt output giữa input thực tế và input nền (reference)
    delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2) :]

    # Tính delta_in: Khác biệt input giữa input thực tế và input nền (reference)
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]

    # dup0 dùng để điều chỉnh shape cho phép toán .repeat() sau đó
    dup0 = [2] + [1 for i in delta_in.shape[1:]]

    grads = [None for _ in grad_input]

    # Tính gradient theo quy tắc DeepLIFT Rescale cho lớp phi tuyến
    # grads[0] = torch.where( condition, value_if_true, value_if_false )

    grads[0] = torch.where(
        # Điều kiện: Nếu delta_in quá nhỏ (tránh chia cho 0)
        torch.abs(delta_in.repeat(dup0)) < 1e-6,
        # Nếu True: Dùng thẳng grad_input (gradient thông thường)
        grad_input[0],
        # Nếu False: Áp dụng quy tắc Rescale của DeepLIFT
        # ===>>> LỖI XẢY RA Ở PHÉP NHÂN TRONG PHẦN NÀY <<<===
        grad_output[0] * (delta_out / delta_in).repeat(dup0)
    )
    return tuple(grads)


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler["Dropout3d"] = passthrough
op_handler["Dropout2d"] = passthrough
op_handler["Dropout"] = passthrough
op_handler["AlphaDropout"] = passthrough

op_handler["Conv1d"] = linear_1d
op_handler["Conv2d"] = linear_1d
op_handler["Conv3d"] = linear_1d
op_handler["ConvTranspose1d"] = linear_1d
op_handler["ConvTranspose2d"] = linear_1d
op_handler["ConvTranspose3d"] = linear_1d
op_handler["Linear"] = linear_1d
op_handler["AvgPool1d"] = linear_1d
op_handler["AvgPool2d"] = linear_1d
op_handler["AvgPool3d"] = linear_1d
op_handler["AdaptiveAvgPool1d"] = linear_1d
op_handler["AdaptiveAvgPool2d"] = linear_1d
op_handler["AdaptiveAvgPool3d"] = linear_1d
op_handler["BatchNorm1d"] = linear_1d
op_handler["BatchNorm2d"] = linear_1d
op_handler["BatchNorm3d"] = linear_1d

op_handler["LeakyReLU"] = nonlinear_1d
op_handler["ReLU"] = nonlinear_1d
op_handler["ELU"] = nonlinear_1d
op_handler["Sigmoid"] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler["Softmax"] = nonlinear_1d
op_handler["SELU"] = nonlinear_1d
op_handler["GELU"] = nonlinear_1d

op_handler["MaxPool1d"] = maxpool
op_handler["MaxPool2d"] = maxpool
op_handler["MaxPool3d"] = maxpool
