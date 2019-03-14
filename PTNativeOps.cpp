#include <ATen/ATen.h>
#include <torch/torch.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void lstm_fwd_func_low_mem_var(
    at::Tensor X,
    at::Tensor W,
    at::Tensor b,
    at::Tensor y0,
    at::Tensor c0,
    at::Tensor i,
    int start,
    int step,
    at::Tensor Y,
    at::Tensor C,
    at::Tensor d,
    at::Tensor x_h,
    at::Tensor intern
);

void lstm_bwd_func_low_mem_var(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i, int start, int step,
    at::Tensor Y, at::Tensor C,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Db, at::Tensor Dh, at::Tensor Dc,
    at::Tensor x_h, at::Tensor intern
);

void lstm_fwd_func_low_mem(
    at::Tensor X,
    at::Tensor W,
    at::Tensor b,
    at::Tensor y0,
    at::Tensor c0,
    at::Tensor i,
    int start,
    int step,
    at::Tensor Y,
    at::Tensor C,
    at::Tensor d
);

void lstm_bwd_func_low_mem(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i, int start, int step,
    at::Tensor Y, at::Tensor C,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Db, at::Tensor Dh, at::Tensor Dc
);

void lstm_fwd_func(
    at::Tensor X, at::Tensor W, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor H, at::Tensor d
);

void lstm_bwd_func(
    at::Tensor X, at::Tensor W, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor H,
    at::Tensor DY, at::Tensor Dd, at::Tensor DX, at::Tensor DW, at::Tensor Dy0, at::Tensor Dc0
);

void lstm_fwd_op_low_mem_var(
    at::Tensor X,
    at::Tensor W,
    at::Tensor b,
    at::Tensor y0,
    at::Tensor c0,
    at::Tensor i,
    int start,
    int step,
    at::Tensor Y,
    at::Tensor C,
    at::Tensor d,
    at::Tensor x_h,
    at::Tensor intern
)
{
    CHECK_INPUT(X);   CHECK_INPUT(W);  CHECK_INPUT(b);
    CHECK_INPUT(y0);  CHECK_INPUT(c0); CHECK_INPUT(i);
    CHECK_INPUT(Y);   CHECK_INPUT(C);  CHECK_INPUT(d);
    CHECK_INPUT(x_h); CHECK_INPUT(intern);
    lstm_fwd_func_low_mem_var(X, W, b, y0, c0, i, start, step, Y, C, d, x_h, intern);
}

void lstm_bwd_op_low_mem_var(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i, int start, int step,
    at::Tensor Y, at::Tensor C,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Db, at::Tensor Dh, at::Tensor Dc,
    at::Tensor x_h, at::Tensor intern
)
{
    CHECK_INPUT(X);   CHECK_INPUT(W);   CHECK_INPUT(b);   CHECK_INPUT(y0);  CHECK_INPUT(c0);  CHECK_INPUT(i);
    CHECK_INPUT(Y);   CHECK_INPUT(C);
    CHECK_INPUT(DY);  CHECK_INPUT(Dd);
    CHECK_INPUT(DX);  CHECK_INPUT(DW);  CHECK_INPUT(Db);  CHECK_INPUT(Dh);  CHECK_INPUT(Dc);
    CHECK_INPUT(x_h); CHECK_INPUT(intern);
    lstm_bwd_func_low_mem_var(X, W, b, y0, c0, i, start, step, Y, C, DY, Dd, DX, DW, Db, Dh, Dc, x_h, intern);
}

void lstm_fwd_op_low_mem(
    at::Tensor X,
    at::Tensor W,
    at::Tensor b,
    at::Tensor y0,
    at::Tensor c0,
    at::Tensor i,
    int start,
    int step,
    at::Tensor Y,
    at::Tensor C,
    at::Tensor d
)
{
    CHECK_INPUT(X);   CHECK_INPUT(W);  CHECK_INPUT(b);
    CHECK_INPUT(y0);  CHECK_INPUT(c0); CHECK_INPUT(i);
    CHECK_INPUT(Y);   CHECK_INPUT(C);  CHECK_INPUT(d);
    lstm_fwd_func_low_mem(X, W, b, y0, c0, i, start, step, Y, C, d);
}

void lstm_bwd_op_low_mem(
    at::Tensor X, at::Tensor W, at::Tensor b, at::Tensor y0, at::Tensor c0, at::Tensor i, int start, int step,
    at::Tensor Y, at::Tensor C,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Db, at::Tensor Dh, at::Tensor Dc
)
{
    CHECK_INPUT(X);   CHECK_INPUT(W);   CHECK_INPUT(b);   CHECK_INPUT(y0);  CHECK_INPUT(c0);  CHECK_INPUT(i);
    CHECK_INPUT(Y);   CHECK_INPUT(C);
    CHECK_INPUT(DY);  CHECK_INPUT(Dd);
    CHECK_INPUT(DX);  CHECK_INPUT(DW);  CHECK_INPUT(Db);  CHECK_INPUT(Dh);  CHECK_INPUT(Dc);
    lstm_bwd_func_low_mem(X, W, b, y0, c0, i, start, step, Y, C, DY, Dd, DX, DW, Db, Dh, Dc);
}

void lstm_fwd_op(
    at::Tensor X, at::Tensor W, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor H, at::Tensor d
)
{
    CHECK_INPUT(X);       CHECK_INPUT(W);     CHECK_INPUT(y0);
    CHECK_INPUT(c0);      CHECK_INPUT(i);
    CHECK_INPUT(Y);       CHECK_INPUT(C);     CHECK_INPUT(H);
    CHECK_INPUT(d);
    lstm_fwd_func(X, W, y0, c0, i, start, step, Y, C, H, d);
}

void lstm_bwd_op(
    at::Tensor X, at::Tensor W, at::Tensor y0, at::Tensor c0, at::Tensor i,
    int start, int step,
    at::Tensor Y, at::Tensor C, at::Tensor H,
    at::Tensor DY, at::Tensor Dd,
    at::Tensor DX, at::Tensor DW, at::Tensor Dy0, at::Tensor Dc0
)
{
    CHECK_INPUT(X);       CHECK_INPUT(W);     CHECK_INPUT(y0);
    CHECK_INPUT(c0);      CHECK_INPUT(i);
    CHECK_INPUT(Y);       CHECK_INPUT(C);     CHECK_INPUT(H);
    CHECK_INPUT(DY);      CHECK_INPUT(Dd);    CHECK_INPUT(DX);
    CHECK_INPUT(DW);      CHECK_INPUT(Dy0);   CHECK_INPUT(Dc0);
    lstm_bwd_func(X, W, y0, c0, i, start, step, Y, C, H, DY, Dd, DX, DW, Dy0, Dc0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lstm_forward_op_low_mem", &lstm_fwd_op_low_mem, "LSTM forward low mem");
  m.def("lstm_backward_op_low_mem", &lstm_bwd_op_low_mem, "LSTM backward low mem");
  m.def("lstm_forward_op_low_mem_var", &lstm_fwd_op_low_mem_var, "LSTM forward low mem var");
  m.def("lstm_backward_op_low_mem_var", &lstm_bwd_op_low_mem_var, "LSTM backward low mem var");
  m.def("lstm_forward_op", &lstm_fwd_op, "LSTM forward");
  m.def("lstm_backward_op", &lstm_bwd_op, "LSTM backward");
}
