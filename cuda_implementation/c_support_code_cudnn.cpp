#include <cudnn.h>
#include <sstream>
#include <string>

using namespace std;

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\\n" << __FILE__ << ':' << __LINE__; \
    std::cerr << _message.str() << "\\nAborting...\\n";                \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                             \
    std::stringstream _error;                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                                \
      _error << "CUDNN failure\\nError: " << cudnnGetErrorString(status);\
      FatalError(_error.str());                                          \
    }                                                                    \
}
