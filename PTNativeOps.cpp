#include <ATen/ATen.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")

void lstm_fwd_func_gpu(
    at::Tensor Z, at::Tensor Wr, at::Tensor c, at::Tensor i,
    at::Tensor Y, at::Tensor d
);

void lstm_bwd_func_gpu(
    at::Tensor Wr, at::Tensor c, at::Tensor i, at::Tensor Y,
    at::Tensor Dd, at::Tensor DZ, at::Tensor DWr, at::Tensor Dc, at::Tensor tmpDc
);

void lstm_fwd_func_cpu(
    at::Tensor Z, at::Tensor Wr, at::Tensor c, at::Tensor i,
    at::Tensor Y, at::Tensor d
);

void lstm_bwd_func_cpu(
    at::Tensor Wr, at::Tensor c, at::Tensor i, at::Tensor Y,
    at::Tensor Dd, at::Tensor DZ, at::Tensor DWr, at::Tensor Dc, at::Tensor tmpDc
);

void lstm_fwd_op_gpu(
    at::Tensor Z, at::Tensor Wr, at::Tensor c, at::Tensor i,
    at::Tensor Y, at::Tensor d
)
{
    CHECK_INPUT(Z);       CHECK_INPUT(Wr);     CHECK_INPUT(c);
    CHECK_INPUT(i);       CHECK_INPUT(Y);       CHECK_INPUT(d);
    lstm_fwd_func_gpu(Z, Wr, c, i, Y, d);
}

void lstm_bwd_op_gpu(
    at::Tensor Wr, at::Tensor c, at::Tensor i, at::Tensor Y,
    at::Tensor Dd, at::Tensor DZ, at::Tensor DWr, at::Tensor Dc, at::Tensor tmpDc
)
{
    CHECK_INPUT(Wr);       CHECK_INPUT(c);     CHECK_INPUT(i);
    CHECK_INPUT(Y);       CHECK_INPUT(Dd);       CHECK_INPUT(DZ);
    CHECK_INPUT(DWr);       CHECK_INPUT(Dc);     CHECK_INPUT(tmpDc);
    lstm_bwd_func_gpu(Wr, c, i, Y, Dd, DZ, DWr, Dc, tmpDc);
}

void lstm_fwd_op_cpu(
    at::Tensor Z, at::Tensor Wr, at::Tensor c, at::Tensor i,
    at::Tensor Y, at::Tensor d
)
{
    CHECK_CPU(Z);      CHECK_CPU(Wr);     CHECK_CPU(c);
    CHECK_CPU(i);      CHECK_CPU(Y);      CHECK_CPU(d);
    lstm_fwd_func_cpu(Z, Wr, c, i, Y, d);
}

void lstm_bwd_op_cpu(
    at::Tensor Wr, at::Tensor c, at::Tensor i, at::Tensor Y,
    at::Tensor Dd, at::Tensor DZ, at::Tensor DWr, at::Tensor Dc, at::Tensor tmpDc
)
{
    CHECK_CPU(Wr);      CHECK_CPU(c);      CHECK_CPU(i);
    CHECK_CPU(Y);       CHECK_CPU(Dd);     CHECK_CPU(DZ);
    CHECK_CPU(DWr);     CHECK_CPU(Dc);     CHECK_CPU(tmpDc);
    lstm_bwd_func_cpu(Wr, c, i, Y, Dd, DZ, DWr, Dc, tmpDc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lstm_forward_op_gpu", &lstm_fwd_op_gpu, "LSTM forward GPU");
  m.def("lstm_backward_op_gpu", &lstm_bwd_op_gpu, "LSTM backward GPU");
  m.def("lstm_forward_op_cpu", &lstm_fwd_op_cpu, "LSTM forward CPU");
  m.def("lstm_backward_op_cpu", &lstm_bwd_op_cpu, "LSTM backward CPU");
}
