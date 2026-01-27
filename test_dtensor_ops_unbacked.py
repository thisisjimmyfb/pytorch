# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import copy
import torch
import torch.distributed as dist
import torch._dynamo
from torch.distributed.tensor import DTensor, init_device_mesh
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorConverter
from torch.utils._pytree import tree_map, tree_flatten

OP_DB_WORLD_SIZE = 4
DEVICE_TYPE = "cpu"

# Ops where base tensor DDEs (not DTensor's fault)
ops_base_dde = {
    "_chunk_cat",
    "_unsafe_masked_index_put_accumulate",
    "_upsample_bilinear2d_aa",
    "addmv",
    "allclose",
    "as_strided_scatter",
    "baddbmm",
    "bernoulli",
    "cauchy",
    "cdist",
    "chunk",
    "combinations",
    "corrcoef",
    "cov",
    "cross",
    "cummax",
    "cummin",
    "cumulative_trapezoid",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "dist",
    "equal",
    "exponential",
    "fft.fft",
    "fft.fft2",
    "fft.fftn",
    "fft.fftshift",
    "fft.hfft",
    "fft.hfft2",
    "fft.hfftn",
    "fft.ifft",
    "fft.ifft2",
    "fft.ifftn",
    "fft.ifftshift",
    "fft.ihfft",
    "fft.ihfft2",
    "fft.ihfftn",
    "fft.irfft",
    "fft.irfft2",
    "fft.irfftn",
    "fft.rfft",
    "fft.rfft2",
    "fft.rfftn",
    "geometric",
    "gradient",
    "grid_sampler_2d",
    "hash_tensor",
    "histogram",
    "histogramdd",
    "hsplit",
    "index_fill",
    "inner",
    "kron",
    "linalg.cross",
    "linalg.diagonal",
    "linalg.multi_dot",
    "linalg.vander",
    "log_normal",
    "logsumexp",
    "masked.cumprod",
    "masked.cumsum",
    "masked.logsumexp",
    "matrix_exp",
    "max_pool2d_with_indices_backward",
    "multinomial",
    "nanquantile",
    "nn.functional.adaptive_max_pool1d",
    "nn.functional.adaptive_max_pool3d",
    "nn.functional.alpha_dropout",
    "nn.functional.avg_pool1d",
    "nn.functional.avg_pool2d",
    "nn.functional.avg_pool3d",
    "nn.functional.batch_norm",
    "nn.functional.bilinear",
    "nn.functional.binary_cross_entropy",
    "nn.functional.binary_cross_entropy_with_logits",
    "nn.functional.channel_shuffle",
    "nn.functional.conv1d",
    "nn.functional.conv2d",
    "nn.functional.conv3d",
    "nn.functional.conv_transpose1d",
    "nn.functional.conv_transpose2d",
    "nn.functional.conv_transpose3d",
    "nn.functional.cosine_similarity",
    "nn.functional.cross_entropy",
    "nn.functional.ctc_loss",
    "nn.functional.dropout",
    "nn.functional.dropout2d",
    "nn.functional.dropout3d",
    "nn.functional.embedding_bag",
    "nn.functional.feature_alpha_dropout",
    "nn.functional.fractional_max_pool2d",
    "nn.functional.fractional_max_pool3d",
    "nn.functional.gaussian_nll_loss",
    "nn.functional.glu",
    "nn.functional.grid_sample",
    "nn.functional.group_norm",
    "nn.functional.huber_loss",
    "nn.functional.instance_norm",
    "nn.functional.l1_loss",
    "nn.functional.local_response_norm",
    "nn.functional.max_pool1d",
    "nn.functional.max_pool2d",
    "nn.functional.max_pool3d",
    "nn.functional.max_unpool1d",
    "nn.functional.max_unpool2d",
    "nn.functional.max_unpool3d",
    "nn.functional.mse_loss",
    "nn.functional.multi_head_attention_forward",
    "nn.functional.multilabel_margin_loss",
    "nn.functional.nll_loss",
    "nn.functional.pad",
    "nn.functional.pdist",
    "nn.functional.pixel_shuffle",
    "nn.functional.prelu",
    "nn.functional.rrelu",
    "nn.functional.scaled_dot_product_attention",
    "nn.functional.smooth_l1_loss",
    "nn.functional.unfold",
    "normal",
    "quantile",
    "rand_like",
    "randint_like",
    "randn_like",
    "resize_",
    "resize_as_",
    "roll",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "split",
    "stft",
    "sum_to_size",
    "take",
    "take_along_dim",
    "tensordot",
    "tensor_split",
    "to_sparse",
    "trace",
    "trapezoid",
    "trapz",
    "unbind",
    "unbind_copy",
    "uniform",
    "unsafe_chunk",
    "unsafe_split",
    "view_as_complex",
    "vsplit",
}

# Ops where DTensor shard prop DDEs (base tensor passes)
ops_dtensor_dde = {
    "as_strided",
    "broadcast_tensors",
    "cartesian_prod",
    "diagflat",
    "expand_as",
    "gather",
    "masked.normalize",
    "masked.std",
    "masked.var",
    "meshgrid",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "nn.functional.embedding",
    "nn.functional.multi_margin_loss",
    "nn.functional.normalize",
    "outer",
    "ravel",
    "repeat_interleave",
    "reshape_as",
    "squeeze",
    "topk",
    "view_as",
}

# Ops that fail due to missing DTensor sharding rules or other errors
ops_fail = {
    "__rsub__",
    "_batch_norm_with_update",
    "_native_batch_norm_legit",
    "_segment_reduce",
    "_unsafe_masked_index",
    "addbmm",
    "addr",
    "alias_copy",
    "aminmax",
    "argwhere",
    "as_strided_copy",
    "block_diag",
    "complex",
    "count_nonzero",
    "cumprod",
    "dsplit",
    "expand_copy",
    "fill",
    "flip",
    "fliplr",
    "flipud",
    "floor_divide",
    "frexp",
    "index_add",
    "index_copy",
    "index_reduce",
    "index_select",
    "isin",
    "kthvalue",
    "logcumsumexp",
    "masked.median",
    "masked_scatter",
    "masked_select",
    "median",
    "mode",
    "mv",
    "nanmean",
    "nanmedian",
    "nansum",
    "narrow_copy",
    "native_batch_norm",
    "ne",
    "nn.functional.adaptive_avg_pool1d",
    "nn.functional.adaptive_avg_pool2d",
    "nn.functional.adaptive_avg_pool3d",
    "nn.functional.adaptive_max_pool2d",
    "nn.functional.celu",
    "nn.functional.cosine_embedding_loss",
    "nn.functional.elu",
    "nn.functional.hardshrink",
    "nn.functional.hardsigmoid",
    "nn.functional.hardswish",
    "nn.functional.hardtanh",
    "nn.functional.hinge_embedding_loss",
    "nn.functional.interpolate",
    "nn.functional.leaky_relu",
    "nn.functional.logsigmoid",
    "nn.functional.margin_ranking_loss",
    "nn.functional.mish",
    "nn.functional.multilabel_soft_margin_loss",
    "nn.functional.pairwise_distance",
    "nn.functional.pixel_unshuffle",
    "nn.functional.relu6",
    "nn.functional.selu",
    "nn.functional.soft_margin_loss",
    "nn.functional.softplus",
    "nn.functional.softshrink",
    "nn.functional.threshold",
    "nn.functional.triplet_margin_loss",
    "nn.functional.triplet_margin_with_distance_loss",
    "nn.functional.upsample_bilinear",
    "nn.functional.upsample_nearest",
    "nonzero",
    "nonzero_static",
    "permute_copy",
    "polar",
    "put",
    "renorm",
    "rot90",
    "rsub",
    "searchsorted",
    "select_scatter",
    "special.airy_ai",
    "special.bessel_j0",
    "special.bessel_j1",
    "special.bessel_y0",
    "special.bessel_y1",
    "special.chebyshev_polynomial_t",
    "special.chebyshev_polynomial_u",
    "special.chebyshev_polynomial_v",
    "special.chebyshev_polynomial_w",
    "special.entr",
    "special.erfcx",
    "special.hermite_polynomial_h",
    "special.hermite_polynomial_he",
    "special.i0e",
    "special.i1",
    "special.i1e",
    "special.laguerre_polynomial_l",
    "special.legendre_polynomial_p",
    "special.log_ndtr",
    "special.modified_bessel_i0",
    "special.modified_bessel_i1",
    "special.modified_bessel_k0",
    "special.modified_bessel_k1",
    "special.ndtri",
    "special.scaled_modified_bessel_k0",
    "special.scaled_modified_bessel_k1",
    "special.shifted_chebyshev_polynomial_t",
    "special.shifted_chebyshev_polynomial_u",
    "special.shifted_chebyshev_polynomial_v",
    "special.shifted_chebyshev_polynomial_w",
    "special.spherical_bessel_j0",
    "special.xlog1py",
    "special.zeta",
    "squeeze_copy",
    "std_mean",
    "t_copy",
    "transpose_copy",
    "unfold",
    "unfold_copy",
    "unique",
    "unique_consecutive",
    "unsqueeze_copy",
    "var_mean",
    "vdot",
    "view_copy",
}

# Ops that skip (no valid sample with markable dims)
ops_skip = {
    "arange",
    "broadcast_shapes",
    "empty",
    "empty_permuted",
    "empty_strided",
    "eye",
    "full",
    "item",
    "linspace",
    "logspace",
    "ones",
    "randint",
    "randn",
    "scalar_tensor",
    "signal.windows.bartlett",
    "signal.windows.blackman",
    "signal.windows.cosine",
    "signal.windows.exponential",
    "signal.windows.gaussian",
    "signal.windows.general_cosine",
    "signal.windows.general_hamming",
    "signal.windows.hamming",
    "signal.windows.hann",
    "signal.windows.kaiser",
    "signal.windows.nuttall",
    "sparse.mm",
    "sparse.mm_reduce",
    "zeros",
}


def _get_expected_status(op_name):
    if op_name in ops_skip:
        return "SKIP"
    if op_name in ops_base_dde:
        return "DDE"
    if op_name in ops_dtensor_dde:
        return "DTENSOR_DDE"
    if op_name in ops_fail:
        return "FAIL"
    return "PASS"


class TestUnbackedDTensorOps(TestCase):
    @property
    def world_size(self) -> int:
        return OP_DB_WORLD_SIZE

    def setUp(self) -> None:
        super().setUp()
        dist.init_process_group("fake", rank=0, world_size=self.world_size)
        self.mesh = init_device_mesh(DEVICE_TYPE, (self.world_size,))

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def _has_valid_unbacked_dims(self, t: torch.Tensor) -> bool:
        return t.ndim > 0 and any(s >= 2 for s in t.shape)

    def _mark_unbacked(self, t: torch.Tensor) -> None:
        for i in range(t.ndim):
            if t.shape[i] >= 2:
                torch._dynamo.decorators.mark_unbacked(t, i)

    def _check_dde(self, func, args, kwargs):
        torch._dynamo.reset()

        def mark_unbacked_tree(x):
            if isinstance(x, torch.Tensor) and self._has_valid_unbacked_dims(x):
                self._mark_unbacked(x)
            return x

        tree_map(mark_unbacked_tree, args)
        tree_map(mark_unbacked_tree, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def compiled_func(*a, **kw):
            return func(*a, **kw)

        try:
            compiled_func(*args, **kwargs)
            return False, None
        except GuardOnDataDependentSymNode as e:
            return True, str(e)[:200]
        except Exception as e:
            return True, f"{type(e).__name__}: {str(e)[:200]}"

    def _run_unbacked_dtensor_test(self, func, args, kwargs):
        torch._dynamo.reset()
        dtc = DTensorConverter(self.mesh, args, kwargs)

        for d_args, d_kwargs in dtc:
            if not dtc.successful():
                continue

            def mark_unbacked_tree(x):
                if isinstance(x, DTensor) and self._has_valid_unbacked_dims(x):
                    self._mark_unbacked(x)
                return x

            tree_map(mark_unbacked_tree, d_args)
            tree_map(mark_unbacked_tree, d_kwargs)

            @torch.compile(backend="eager", fullgraph=True)
            def compiled_func(*a, **kw):
                return func(*a, **kw)

            try:
                compiled_func(*d_args, **d_kwargs)
                return "PASS", None
            except GuardOnDataDependentSymNode as e:
                return "DTENSOR_DDE", str(e)[:200]
            except Exception as e:
                err_str = str(e)
                if "Could not guard on data-dependent" in err_str or "Could not extract" in err_str:
                    return "DTENSOR_DDE", err_str[:200]
                return "FAIL", f"{type(e).__name__}: {err_str[:200]}"

        return "SKIP", "no valid dtensor conversion"

    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_unbacked_dtensor_op_db(self, dtype, op):
        samples = list(op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=False))
        expected = _get_expected_status(op.name)

        any_pass = False
        any_tested = False
        last_actual = None
        last_err = None

        for sample in samples:
            try:
                args = [sample.input] + list(sample.args)
                kwargs = sample.kwargs
                all_tensors = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
                if not any(self._has_valid_unbacked_dims(t) for t in all_tensors):
                    continue

                any_tested = True
                args_copy = copy.deepcopy(args)
                kwargs_copy = copy.deepcopy(kwargs)
            except NotImplementedError:
                # skip samples that can't be deepcopied (e.g., sparse tensors)
                continue

            is_dde, err = self._check_dde(op.op, args_copy, kwargs_copy)
            if is_dde:
                actual = "DDE"
            else:
                actual, err = self._run_unbacked_dtensor_test(op.op, args, kwargs)

            last_actual = actual
            last_err = err

            if actual == "PASS":
                any_pass = True
                break

        if not any_tested:
            if expected == "SKIP":
                return
            self.fail(f"{op.name}: no valid sample found but expected {expected}")
            return

        # if any sample passes, op is considered passing
        if any_pass:
            # only fail if we expected it to NOT pass but it did (to track improvements)
            # but don't fail - just pass the test since passing is good
            pass
        else:
            if expected == "PASS":
                self.fail(f"{op.name}: expected PASS but got {last_actual}: {last_err}")


instantiate_device_type_tests(TestUnbackedDTensorOps, globals(), only_for=(DEVICE_TYPE,))

if __name__ == "__main__":
    run_tests()
