#ifndef TRT_BEV_POOL_H
#define TRT_BEV_POOL_H

#include <string>
#include <vector>

#include "NvInferPlugin.h"

namespace nvinfer1 {
namespace plugin {
class TRTBEVPoolV2 : public IPluginV2DynamicExt {
 public:
  TRTBEVPoolV2(const std::string &name, int outWidth, int outHeight);

  TRTBEVPoolV2(const std::string name, const void *data, size_t length);

  TRTBEVPoolV2() = delete;

  ~TRTBEVPoolV2() noexcept override = default;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs,
                                 int nbOutputs) noexcept override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char *getPluginType() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  int getNbOutputs() const noexcept override;

  size_t getSerializationSize() const noexcept override;

  void serialize(void *buffer) const noexcept override;

  int32_t initialize() noexcept override;

  void terminate() noexcept override;

  void destroy() noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override;

  const char *getPluginNamespace() const noexcept override;

 protected:
  int mOutWidth;
  int mOutHeight;
  std::string mLayerName;
  std::string mPluginNamespace;
};

class TRTBEVPoolV2Creator : public IPluginCreator {
 public:
  TRTBEVPoolV2Creator();

  ~TRTBEVPoolV2Creator() noexcept override = default;

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const PluginFieldCollection *getFieldNames() noexcept override;

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                                         size_t serialLength) noexcept override;

  void setPluginNamespace(char const *libNamespace) noexcept override;

  char const *getPluginNamespace() const noexcept override;

 protected:
  std::string mNamespace;
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mPluginName;
  std::string mPluginVersion;
};
}  // namespace plugin
}  // namespace nvinfer1
#endif  // TRT_BEV_POOL_H
