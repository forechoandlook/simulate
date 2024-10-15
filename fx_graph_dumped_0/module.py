
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.load_state_dict(torch.load(r'fx_graph_dumped_0/state_dict.pt'))

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321):
        convolution = torch.ops.aten.convolution.default(primals_321, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        add = torch.ops.aten.add.Tensor(primals_164, 1);  primals_164 = None
        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, primals_2, primals_3, primals_162, primals_163, True, 0.1, 1e-05);  primals_3 = primals_162 = primals_163 = None
        getitem = _native_batch_norm_legit_functional[0]
        getitem_1 = _native_batch_norm_legit_functional[1]
        getitem_2 = _native_batch_norm_legit_functional[2]
        getitem_3 = _native_batch_norm_legit_functional[3]
        getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
        relu = torch.ops.aten.relu.default(getitem);  getitem = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
        getitem_5 = max_pool2d_with_indices[0]
        getitem_6 = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        convolution_1 = torch.ops.aten.convolution.default(getitem_5, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_1 = torch.ops.aten.add.Tensor(primals_167, 1);  primals_167 = None
        _native_batch_norm_legit_functional_1 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_1, primals_5, primals_6, primals_165, primals_166, True, 0.1, 1e-05);  primals_6 = primals_165 = primals_166 = None
        getitem_7 = _native_batch_norm_legit_functional_1[0]
        getitem_8 = _native_batch_norm_legit_functional_1[1]
        getitem_9 = _native_batch_norm_legit_functional_1[2]
        getitem_10 = _native_batch_norm_legit_functional_1[3]
        getitem_11 = _native_batch_norm_legit_functional_1[4];  _native_batch_norm_legit_functional_1 = None
        relu_1 = torch.ops.aten.relu.default(getitem_7);  getitem_7 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_2 = torch.ops.aten.add.Tensor(primals_170, 1);  primals_170 = None
        _native_batch_norm_legit_functional_2 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_2, primals_8, primals_9, primals_168, primals_169, True, 0.1, 1e-05);  primals_9 = primals_168 = primals_169 = None
        getitem_12 = _native_batch_norm_legit_functional_2[0]
        getitem_13 = _native_batch_norm_legit_functional_2[1]
        getitem_14 = _native_batch_norm_legit_functional_2[2]
        getitem_15 = _native_batch_norm_legit_functional_2[3]
        getitem_16 = _native_batch_norm_legit_functional_2[4];  _native_batch_norm_legit_functional_2 = None
        relu_2 = torch.ops.aten.relu.default(getitem_12);  getitem_12 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_3 = torch.ops.aten.add.Tensor(primals_173, 1);  primals_173 = None
        _native_batch_norm_legit_functional_3 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_3, primals_11, primals_12, primals_171, primals_172, True, 0.1, 1e-05);  primals_12 = primals_171 = primals_172 = None
        getitem_17 = _native_batch_norm_legit_functional_3[0]
        getitem_18 = _native_batch_norm_legit_functional_3[1]
        getitem_19 = _native_batch_norm_legit_functional_3[2]
        getitem_20 = _native_batch_norm_legit_functional_3[3]
        getitem_21 = _native_batch_norm_legit_functional_3[4];  _native_batch_norm_legit_functional_3 = None
        convolution_4 = torch.ops.aten.convolution.default(getitem_5, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_4 = torch.ops.aten.add.Tensor(primals_176, 1);  primals_176 = None
        _native_batch_norm_legit_functional_4 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_4, primals_14, primals_15, primals_174, primals_175, True, 0.1, 1e-05);  primals_15 = primals_174 = primals_175 = None
        getitem_22 = _native_batch_norm_legit_functional_4[0]
        getitem_23 = _native_batch_norm_legit_functional_4[1]
        getitem_24 = _native_batch_norm_legit_functional_4[2]
        getitem_25 = _native_batch_norm_legit_functional_4[3]
        getitem_26 = _native_batch_norm_legit_functional_4[4];  _native_batch_norm_legit_functional_4 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_17, getitem_22);  getitem_17 = getitem_22 = None
        relu_3 = torch.ops.aten.relu.default(add_5);  add_5 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_6 = torch.ops.aten.add.Tensor(primals_179, 1);  primals_179 = None
        _native_batch_norm_legit_functional_5 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_5, primals_17, primals_18, primals_177, primals_178, True, 0.1, 1e-05);  primals_18 = primals_177 = primals_178 = None
        getitem_27 = _native_batch_norm_legit_functional_5[0]
        getitem_28 = _native_batch_norm_legit_functional_5[1]
        getitem_29 = _native_batch_norm_legit_functional_5[2]
        getitem_30 = _native_batch_norm_legit_functional_5[3]
        getitem_31 = _native_batch_norm_legit_functional_5[4];  _native_batch_norm_legit_functional_5 = None
        relu_4 = torch.ops.aten.relu.default(getitem_27);  getitem_27 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_4, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_7 = torch.ops.aten.add.Tensor(primals_182, 1);  primals_182 = None
        _native_batch_norm_legit_functional_6 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_6, primals_20, primals_21, primals_180, primals_181, True, 0.1, 1e-05);  primals_21 = primals_180 = primals_181 = None
        getitem_32 = _native_batch_norm_legit_functional_6[0]
        getitem_33 = _native_batch_norm_legit_functional_6[1]
        getitem_34 = _native_batch_norm_legit_functional_6[2]
        getitem_35 = _native_batch_norm_legit_functional_6[3]
        getitem_36 = _native_batch_norm_legit_functional_6[4];  _native_batch_norm_legit_functional_6 = None
        relu_5 = torch.ops.aten.relu.default(getitem_32);  getitem_32 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_8 = torch.ops.aten.add.Tensor(primals_185, 1);  primals_185 = None
        _native_batch_norm_legit_functional_7 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_7, primals_23, primals_24, primals_183, primals_184, True, 0.1, 1e-05);  primals_24 = primals_183 = primals_184 = None
        getitem_37 = _native_batch_norm_legit_functional_7[0]
        getitem_38 = _native_batch_norm_legit_functional_7[1]
        getitem_39 = _native_batch_norm_legit_functional_7[2]
        getitem_40 = _native_batch_norm_legit_functional_7[3]
        getitem_41 = _native_batch_norm_legit_functional_7[4];  _native_batch_norm_legit_functional_7 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_37, relu_3);  getitem_37 = None
        relu_6 = torch.ops.aten.relu.default(add_9);  add_9 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_10 = torch.ops.aten.add.Tensor(primals_188, 1);  primals_188 = None
        _native_batch_norm_legit_functional_8 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_8, primals_26, primals_27, primals_186, primals_187, True, 0.1, 1e-05);  primals_27 = primals_186 = primals_187 = None
        getitem_42 = _native_batch_norm_legit_functional_8[0]
        getitem_43 = _native_batch_norm_legit_functional_8[1]
        getitem_44 = _native_batch_norm_legit_functional_8[2]
        getitem_45 = _native_batch_norm_legit_functional_8[3]
        getitem_46 = _native_batch_norm_legit_functional_8[4];  _native_batch_norm_legit_functional_8 = None
        relu_7 = torch.ops.aten.relu.default(getitem_42);  getitem_42 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_11 = torch.ops.aten.add.Tensor(primals_191, 1);  primals_191 = None
        _native_batch_norm_legit_functional_9 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_9, primals_29, primals_30, primals_189, primals_190, True, 0.1, 1e-05);  primals_30 = primals_189 = primals_190 = None
        getitem_47 = _native_batch_norm_legit_functional_9[0]
        getitem_48 = _native_batch_norm_legit_functional_9[1]
        getitem_49 = _native_batch_norm_legit_functional_9[2]
        getitem_50 = _native_batch_norm_legit_functional_9[3]
        getitem_51 = _native_batch_norm_legit_functional_9[4];  _native_batch_norm_legit_functional_9 = None
        relu_8 = torch.ops.aten.relu.default(getitem_47);  getitem_47 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_8, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_12 = torch.ops.aten.add.Tensor(primals_194, 1);  primals_194 = None
        _native_batch_norm_legit_functional_10 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_10, primals_32, primals_33, primals_192, primals_193, True, 0.1, 1e-05);  primals_33 = primals_192 = primals_193 = None
        getitem_52 = _native_batch_norm_legit_functional_10[0]
        getitem_53 = _native_batch_norm_legit_functional_10[1]
        getitem_54 = _native_batch_norm_legit_functional_10[2]
        getitem_55 = _native_batch_norm_legit_functional_10[3]
        getitem_56 = _native_batch_norm_legit_functional_10[4];  _native_batch_norm_legit_functional_10 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_52, relu_6);  getitem_52 = None
        relu_9 = torch.ops.aten.relu.default(add_13);  add_13 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_9, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_14 = torch.ops.aten.add.Tensor(primals_197, 1);  primals_197 = None
        _native_batch_norm_legit_functional_11 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_11, primals_35, primals_36, primals_195, primals_196, True, 0.1, 1e-05);  primals_36 = primals_195 = primals_196 = None
        getitem_57 = _native_batch_norm_legit_functional_11[0]
        getitem_58 = _native_batch_norm_legit_functional_11[1]
        getitem_59 = _native_batch_norm_legit_functional_11[2]
        getitem_60 = _native_batch_norm_legit_functional_11[3]
        getitem_61 = _native_batch_norm_legit_functional_11[4];  _native_batch_norm_legit_functional_11 = None
        relu_10 = torch.ops.aten.relu.default(getitem_57);  getitem_57 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_10, primals_37, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_15 = torch.ops.aten.add.Tensor(primals_200, 1);  primals_200 = None
        _native_batch_norm_legit_functional_12 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_12, primals_38, primals_39, primals_198, primals_199, True, 0.1, 1e-05);  primals_39 = primals_198 = primals_199 = None
        getitem_62 = _native_batch_norm_legit_functional_12[0]
        getitem_63 = _native_batch_norm_legit_functional_12[1]
        getitem_64 = _native_batch_norm_legit_functional_12[2]
        getitem_65 = _native_batch_norm_legit_functional_12[3]
        getitem_66 = _native_batch_norm_legit_functional_12[4];  _native_batch_norm_legit_functional_12 = None
        relu_11 = torch.ops.aten.relu.default(getitem_62);  getitem_62 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_11, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_16 = torch.ops.aten.add.Tensor(primals_203, 1);  primals_203 = None
        _native_batch_norm_legit_functional_13 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_13, primals_41, primals_42, primals_201, primals_202, True, 0.1, 1e-05);  primals_42 = primals_201 = primals_202 = None
        getitem_67 = _native_batch_norm_legit_functional_13[0]
        getitem_68 = _native_batch_norm_legit_functional_13[1]
        getitem_69 = _native_batch_norm_legit_functional_13[2]
        getitem_70 = _native_batch_norm_legit_functional_13[3]
        getitem_71 = _native_batch_norm_legit_functional_13[4];  _native_batch_norm_legit_functional_13 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_9, primals_43, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_17 = torch.ops.aten.add.Tensor(primals_206, 1);  primals_206 = None
        _native_batch_norm_legit_functional_14 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_14, primals_44, primals_45, primals_204, primals_205, True, 0.1, 1e-05);  primals_45 = primals_204 = primals_205 = None
        getitem_72 = _native_batch_norm_legit_functional_14[0]
        getitem_73 = _native_batch_norm_legit_functional_14[1]
        getitem_74 = _native_batch_norm_legit_functional_14[2]
        getitem_75 = _native_batch_norm_legit_functional_14[3]
        getitem_76 = _native_batch_norm_legit_functional_14[4];  _native_batch_norm_legit_functional_14 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_67, getitem_72);  getitem_67 = getitem_72 = None
        relu_12 = torch.ops.aten.relu.default(add_18);  add_18 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_12, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_19 = torch.ops.aten.add.Tensor(primals_209, 1);  primals_209 = None
        _native_batch_norm_legit_functional_15 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_15, primals_47, primals_48, primals_207, primals_208, True, 0.1, 1e-05);  primals_48 = primals_207 = primals_208 = None
        getitem_77 = _native_batch_norm_legit_functional_15[0]
        getitem_78 = _native_batch_norm_legit_functional_15[1]
        getitem_79 = _native_batch_norm_legit_functional_15[2]
        getitem_80 = _native_batch_norm_legit_functional_15[3]
        getitem_81 = _native_batch_norm_legit_functional_15[4];  _native_batch_norm_legit_functional_15 = None
        relu_13 = torch.ops.aten.relu.default(getitem_77);  getitem_77 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_20 = torch.ops.aten.add.Tensor(primals_212, 1);  primals_212 = None
        _native_batch_norm_legit_functional_16 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_16, primals_50, primals_51, primals_210, primals_211, True, 0.1, 1e-05);  primals_51 = primals_210 = primals_211 = None
        getitem_82 = _native_batch_norm_legit_functional_16[0]
        getitem_83 = _native_batch_norm_legit_functional_16[1]
        getitem_84 = _native_batch_norm_legit_functional_16[2]
        getitem_85 = _native_batch_norm_legit_functional_16[3]
        getitem_86 = _native_batch_norm_legit_functional_16[4];  _native_batch_norm_legit_functional_16 = None
        relu_14 = torch.ops.aten.relu.default(getitem_82);  getitem_82 = None
        convolution_17 = torch.ops.aten.convolution.default(relu_14, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_21 = torch.ops.aten.add.Tensor(primals_215, 1);  primals_215 = None
        _native_batch_norm_legit_functional_17 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_17, primals_53, primals_54, primals_213, primals_214, True, 0.1, 1e-05);  primals_54 = primals_213 = primals_214 = None
        getitem_87 = _native_batch_norm_legit_functional_17[0]
        getitem_88 = _native_batch_norm_legit_functional_17[1]
        getitem_89 = _native_batch_norm_legit_functional_17[2]
        getitem_90 = _native_batch_norm_legit_functional_17[3]
        getitem_91 = _native_batch_norm_legit_functional_17[4];  _native_batch_norm_legit_functional_17 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_87, relu_12);  getitem_87 = None
        relu_15 = torch.ops.aten.relu.default(add_22);  add_22 = None
        convolution_18 = torch.ops.aten.convolution.default(relu_15, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_23 = torch.ops.aten.add.Tensor(primals_218, 1);  primals_218 = None
        _native_batch_norm_legit_functional_18 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_18, primals_56, primals_57, primals_216, primals_217, True, 0.1, 1e-05);  primals_57 = primals_216 = primals_217 = None
        getitem_92 = _native_batch_norm_legit_functional_18[0]
        getitem_93 = _native_batch_norm_legit_functional_18[1]
        getitem_94 = _native_batch_norm_legit_functional_18[2]
        getitem_95 = _native_batch_norm_legit_functional_18[3]
        getitem_96 = _native_batch_norm_legit_functional_18[4];  _native_batch_norm_legit_functional_18 = None
        relu_16 = torch.ops.aten.relu.default(getitem_92);  getitem_92 = None
        convolution_19 = torch.ops.aten.convolution.default(relu_16, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_24 = torch.ops.aten.add.Tensor(primals_221, 1);  primals_221 = None
        _native_batch_norm_legit_functional_19 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_19, primals_59, primals_60, primals_219, primals_220, True, 0.1, 1e-05);  primals_60 = primals_219 = primals_220 = None
        getitem_97 = _native_batch_norm_legit_functional_19[0]
        getitem_98 = _native_batch_norm_legit_functional_19[1]
        getitem_99 = _native_batch_norm_legit_functional_19[2]
        getitem_100 = _native_batch_norm_legit_functional_19[3]
        getitem_101 = _native_batch_norm_legit_functional_19[4];  _native_batch_norm_legit_functional_19 = None
        relu_17 = torch.ops.aten.relu.default(getitem_97);  getitem_97 = None
        convolution_20 = torch.ops.aten.convolution.default(relu_17, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_25 = torch.ops.aten.add.Tensor(primals_224, 1);  primals_224 = None
        _native_batch_norm_legit_functional_20 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_20, primals_62, primals_63, primals_222, primals_223, True, 0.1, 1e-05);  primals_63 = primals_222 = primals_223 = None
        getitem_102 = _native_batch_norm_legit_functional_20[0]
        getitem_103 = _native_batch_norm_legit_functional_20[1]
        getitem_104 = _native_batch_norm_legit_functional_20[2]
        getitem_105 = _native_batch_norm_legit_functional_20[3]
        getitem_106 = _native_batch_norm_legit_functional_20[4];  _native_batch_norm_legit_functional_20 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_102, relu_15);  getitem_102 = None
        relu_18 = torch.ops.aten.relu.default(add_26);  add_26 = None
        convolution_21 = torch.ops.aten.convolution.default(relu_18, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_27 = torch.ops.aten.add.Tensor(primals_227, 1);  primals_227 = None
        _native_batch_norm_legit_functional_21 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_21, primals_65, primals_66, primals_225, primals_226, True, 0.1, 1e-05);  primals_66 = primals_225 = primals_226 = None
        getitem_107 = _native_batch_norm_legit_functional_21[0]
        getitem_108 = _native_batch_norm_legit_functional_21[1]
        getitem_109 = _native_batch_norm_legit_functional_21[2]
        getitem_110 = _native_batch_norm_legit_functional_21[3]
        getitem_111 = _native_batch_norm_legit_functional_21[4];  _native_batch_norm_legit_functional_21 = None
        relu_19 = torch.ops.aten.relu.default(getitem_107);  getitem_107 = None
        convolution_22 = torch.ops.aten.convolution.default(relu_19, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_28 = torch.ops.aten.add.Tensor(primals_230, 1);  primals_230 = None
        _native_batch_norm_legit_functional_22 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_22, primals_68, primals_69, primals_228, primals_229, True, 0.1, 1e-05);  primals_69 = primals_228 = primals_229 = None
        getitem_112 = _native_batch_norm_legit_functional_22[0]
        getitem_113 = _native_batch_norm_legit_functional_22[1]
        getitem_114 = _native_batch_norm_legit_functional_22[2]
        getitem_115 = _native_batch_norm_legit_functional_22[3]
        getitem_116 = _native_batch_norm_legit_functional_22[4];  _native_batch_norm_legit_functional_22 = None
        relu_20 = torch.ops.aten.relu.default(getitem_112);  getitem_112 = None
        convolution_23 = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_29 = torch.ops.aten.add.Tensor(primals_233, 1);  primals_233 = None
        _native_batch_norm_legit_functional_23 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_23, primals_71, primals_72, primals_231, primals_232, True, 0.1, 1e-05);  primals_72 = primals_231 = primals_232 = None
        getitem_117 = _native_batch_norm_legit_functional_23[0]
        getitem_118 = _native_batch_norm_legit_functional_23[1]
        getitem_119 = _native_batch_norm_legit_functional_23[2]
        getitem_120 = _native_batch_norm_legit_functional_23[3]
        getitem_121 = _native_batch_norm_legit_functional_23[4];  _native_batch_norm_legit_functional_23 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_117, relu_18);  getitem_117 = None
        relu_21 = torch.ops.aten.relu.default(add_30);  add_30 = None
        convolution_24 = torch.ops.aten.convolution.default(relu_21, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_31 = torch.ops.aten.add.Tensor(primals_236, 1);  primals_236 = None
        _native_batch_norm_legit_functional_24 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_24, primals_74, primals_75, primals_234, primals_235, True, 0.1, 1e-05);  primals_75 = primals_234 = primals_235 = None
        getitem_122 = _native_batch_norm_legit_functional_24[0]
        getitem_123 = _native_batch_norm_legit_functional_24[1]
        getitem_124 = _native_batch_norm_legit_functional_24[2]
        getitem_125 = _native_batch_norm_legit_functional_24[3]
        getitem_126 = _native_batch_norm_legit_functional_24[4];  _native_batch_norm_legit_functional_24 = None
        relu_22 = torch.ops.aten.relu.default(getitem_122);  getitem_122 = None
        convolution_25 = torch.ops.aten.convolution.default(relu_22, primals_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_32 = torch.ops.aten.add.Tensor(primals_239, 1);  primals_239 = None
        _native_batch_norm_legit_functional_25 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_25, primals_77, primals_78, primals_237, primals_238, True, 0.1, 1e-05);  primals_78 = primals_237 = primals_238 = None
        getitem_127 = _native_batch_norm_legit_functional_25[0]
        getitem_128 = _native_batch_norm_legit_functional_25[1]
        getitem_129 = _native_batch_norm_legit_functional_25[2]
        getitem_130 = _native_batch_norm_legit_functional_25[3]
        getitem_131 = _native_batch_norm_legit_functional_25[4];  _native_batch_norm_legit_functional_25 = None
        relu_23 = torch.ops.aten.relu.default(getitem_127);  getitem_127 = None
        convolution_26 = torch.ops.aten.convolution.default(relu_23, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_33 = torch.ops.aten.add.Tensor(primals_242, 1);  primals_242 = None
        _native_batch_norm_legit_functional_26 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_26, primals_80, primals_81, primals_240, primals_241, True, 0.1, 1e-05);  primals_81 = primals_240 = primals_241 = None
        getitem_132 = _native_batch_norm_legit_functional_26[0]
        getitem_133 = _native_batch_norm_legit_functional_26[1]
        getitem_134 = _native_batch_norm_legit_functional_26[2]
        getitem_135 = _native_batch_norm_legit_functional_26[3]
        getitem_136 = _native_batch_norm_legit_functional_26[4];  _native_batch_norm_legit_functional_26 = None
        convolution_27 = torch.ops.aten.convolution.default(relu_21, primals_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_34 = torch.ops.aten.add.Tensor(primals_245, 1);  primals_245 = None
        _native_batch_norm_legit_functional_27 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_27, primals_83, primals_84, primals_243, primals_244, True, 0.1, 1e-05);  primals_84 = primals_243 = primals_244 = None
        getitem_137 = _native_batch_norm_legit_functional_27[0]
        getitem_138 = _native_batch_norm_legit_functional_27[1]
        getitem_139 = _native_batch_norm_legit_functional_27[2]
        getitem_140 = _native_batch_norm_legit_functional_27[3]
        getitem_141 = _native_batch_norm_legit_functional_27[4];  _native_batch_norm_legit_functional_27 = None
        add_35 = torch.ops.aten.add.Tensor(getitem_132, getitem_137);  getitem_132 = getitem_137 = None
        relu_24 = torch.ops.aten.relu.default(add_35);  add_35 = None
        convolution_28 = torch.ops.aten.convolution.default(relu_24, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_36 = torch.ops.aten.add.Tensor(primals_248, 1);  primals_248 = None
        _native_batch_norm_legit_functional_28 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_28, primals_86, primals_87, primals_246, primals_247, True, 0.1, 1e-05);  primals_87 = primals_246 = primals_247 = None
        getitem_142 = _native_batch_norm_legit_functional_28[0]
        getitem_143 = _native_batch_norm_legit_functional_28[1]
        getitem_144 = _native_batch_norm_legit_functional_28[2]
        getitem_145 = _native_batch_norm_legit_functional_28[3]
        getitem_146 = _native_batch_norm_legit_functional_28[4];  _native_batch_norm_legit_functional_28 = None
        relu_25 = torch.ops.aten.relu.default(getitem_142);  getitem_142 = None
        convolution_29 = torch.ops.aten.convolution.default(relu_25, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_37 = torch.ops.aten.add.Tensor(primals_251, 1);  primals_251 = None
        _native_batch_norm_legit_functional_29 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_29, primals_89, primals_90, primals_249, primals_250, True, 0.1, 1e-05);  primals_90 = primals_249 = primals_250 = None
        getitem_147 = _native_batch_norm_legit_functional_29[0]
        getitem_148 = _native_batch_norm_legit_functional_29[1]
        getitem_149 = _native_batch_norm_legit_functional_29[2]
        getitem_150 = _native_batch_norm_legit_functional_29[3]
        getitem_151 = _native_batch_norm_legit_functional_29[4];  _native_batch_norm_legit_functional_29 = None
        relu_26 = torch.ops.aten.relu.default(getitem_147);  getitem_147 = None
        convolution_30 = torch.ops.aten.convolution.default(relu_26, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_38 = torch.ops.aten.add.Tensor(primals_254, 1);  primals_254 = None
        _native_batch_norm_legit_functional_30 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_30, primals_92, primals_93, primals_252, primals_253, True, 0.1, 1e-05);  primals_93 = primals_252 = primals_253 = None
        getitem_152 = _native_batch_norm_legit_functional_30[0]
        getitem_153 = _native_batch_norm_legit_functional_30[1]
        getitem_154 = _native_batch_norm_legit_functional_30[2]
        getitem_155 = _native_batch_norm_legit_functional_30[3]
        getitem_156 = _native_batch_norm_legit_functional_30[4];  _native_batch_norm_legit_functional_30 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_152, relu_24);  getitem_152 = None
        relu_27 = torch.ops.aten.relu.default(add_39);  add_39 = None
        convolution_31 = torch.ops.aten.convolution.default(relu_27, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_40 = torch.ops.aten.add.Tensor(primals_257, 1);  primals_257 = None
        _native_batch_norm_legit_functional_31 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_31, primals_95, primals_96, primals_255, primals_256, True, 0.1, 1e-05);  primals_96 = primals_255 = primals_256 = None
        getitem_157 = _native_batch_norm_legit_functional_31[0]
        getitem_158 = _native_batch_norm_legit_functional_31[1]
        getitem_159 = _native_batch_norm_legit_functional_31[2]
        getitem_160 = _native_batch_norm_legit_functional_31[3]
        getitem_161 = _native_batch_norm_legit_functional_31[4];  _native_batch_norm_legit_functional_31 = None
        relu_28 = torch.ops.aten.relu.default(getitem_157);  getitem_157 = None
        convolution_32 = torch.ops.aten.convolution.default(relu_28, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_41 = torch.ops.aten.add.Tensor(primals_260, 1);  primals_260 = None
        _native_batch_norm_legit_functional_32 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_32, primals_98, primals_99, primals_258, primals_259, True, 0.1, 1e-05);  primals_99 = primals_258 = primals_259 = None
        getitem_162 = _native_batch_norm_legit_functional_32[0]
        getitem_163 = _native_batch_norm_legit_functional_32[1]
        getitem_164 = _native_batch_norm_legit_functional_32[2]
        getitem_165 = _native_batch_norm_legit_functional_32[3]
        getitem_166 = _native_batch_norm_legit_functional_32[4];  _native_batch_norm_legit_functional_32 = None
        relu_29 = torch.ops.aten.relu.default(getitem_162);  getitem_162 = None
        convolution_33 = torch.ops.aten.convolution.default(relu_29, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_42 = torch.ops.aten.add.Tensor(primals_263, 1);  primals_263 = None
        _native_batch_norm_legit_functional_33 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_33, primals_101, primals_102, primals_261, primals_262, True, 0.1, 1e-05);  primals_102 = primals_261 = primals_262 = None
        getitem_167 = _native_batch_norm_legit_functional_33[0]
        getitem_168 = _native_batch_norm_legit_functional_33[1]
        getitem_169 = _native_batch_norm_legit_functional_33[2]
        getitem_170 = _native_batch_norm_legit_functional_33[3]
        getitem_171 = _native_batch_norm_legit_functional_33[4];  _native_batch_norm_legit_functional_33 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_167, relu_27);  getitem_167 = None
        relu_30 = torch.ops.aten.relu.default(add_43);  add_43 = None
        convolution_34 = torch.ops.aten.convolution.default(relu_30, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_44 = torch.ops.aten.add.Tensor(primals_266, 1);  primals_266 = None
        _native_batch_norm_legit_functional_34 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_34, primals_104, primals_105, primals_264, primals_265, True, 0.1, 1e-05);  primals_105 = primals_264 = primals_265 = None
        getitem_172 = _native_batch_norm_legit_functional_34[0]
        getitem_173 = _native_batch_norm_legit_functional_34[1]
        getitem_174 = _native_batch_norm_legit_functional_34[2]
        getitem_175 = _native_batch_norm_legit_functional_34[3]
        getitem_176 = _native_batch_norm_legit_functional_34[4];  _native_batch_norm_legit_functional_34 = None
        relu_31 = torch.ops.aten.relu.default(getitem_172);  getitem_172 = None
        convolution_35 = torch.ops.aten.convolution.default(relu_31, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_45 = torch.ops.aten.add.Tensor(primals_269, 1);  primals_269 = None
        _native_batch_norm_legit_functional_35 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_35, primals_107, primals_108, primals_267, primals_268, True, 0.1, 1e-05);  primals_108 = primals_267 = primals_268 = None
        getitem_177 = _native_batch_norm_legit_functional_35[0]
        getitem_178 = _native_batch_norm_legit_functional_35[1]
        getitem_179 = _native_batch_norm_legit_functional_35[2]
        getitem_180 = _native_batch_norm_legit_functional_35[3]
        getitem_181 = _native_batch_norm_legit_functional_35[4];  _native_batch_norm_legit_functional_35 = None
        relu_32 = torch.ops.aten.relu.default(getitem_177);  getitem_177 = None
        convolution_36 = torch.ops.aten.convolution.default(relu_32, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_46 = torch.ops.aten.add.Tensor(primals_272, 1);  primals_272 = None
        _native_batch_norm_legit_functional_36 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_36, primals_110, primals_111, primals_270, primals_271, True, 0.1, 1e-05);  primals_111 = primals_270 = primals_271 = None
        getitem_182 = _native_batch_norm_legit_functional_36[0]
        getitem_183 = _native_batch_norm_legit_functional_36[1]
        getitem_184 = _native_batch_norm_legit_functional_36[2]
        getitem_185 = _native_batch_norm_legit_functional_36[3]
        getitem_186 = _native_batch_norm_legit_functional_36[4];  _native_batch_norm_legit_functional_36 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_182, relu_30);  getitem_182 = None
        relu_33 = torch.ops.aten.relu.default(add_47);  add_47 = None
        convolution_37 = torch.ops.aten.convolution.default(relu_33, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_48 = torch.ops.aten.add.Tensor(primals_275, 1);  primals_275 = None
        _native_batch_norm_legit_functional_37 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_37, primals_113, primals_114, primals_273, primals_274, True, 0.1, 1e-05);  primals_114 = primals_273 = primals_274 = None
        getitem_187 = _native_batch_norm_legit_functional_37[0]
        getitem_188 = _native_batch_norm_legit_functional_37[1]
        getitem_189 = _native_batch_norm_legit_functional_37[2]
        getitem_190 = _native_batch_norm_legit_functional_37[3]
        getitem_191 = _native_batch_norm_legit_functional_37[4];  _native_batch_norm_legit_functional_37 = None
        relu_34 = torch.ops.aten.relu.default(getitem_187);  getitem_187 = None
        convolution_38 = torch.ops.aten.convolution.default(relu_34, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_49 = torch.ops.aten.add.Tensor(primals_278, 1);  primals_278 = None
        _native_batch_norm_legit_functional_38 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_38, primals_116, primals_117, primals_276, primals_277, True, 0.1, 1e-05);  primals_117 = primals_276 = primals_277 = None
        getitem_192 = _native_batch_norm_legit_functional_38[0]
        getitem_193 = _native_batch_norm_legit_functional_38[1]
        getitem_194 = _native_batch_norm_legit_functional_38[2]
        getitem_195 = _native_batch_norm_legit_functional_38[3]
        getitem_196 = _native_batch_norm_legit_functional_38[4];  _native_batch_norm_legit_functional_38 = None
        relu_35 = torch.ops.aten.relu.default(getitem_192);  getitem_192 = None
        convolution_39 = torch.ops.aten.convolution.default(relu_35, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_50 = torch.ops.aten.add.Tensor(primals_281, 1);  primals_281 = None
        _native_batch_norm_legit_functional_39 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_39, primals_119, primals_120, primals_279, primals_280, True, 0.1, 1e-05);  primals_120 = primals_279 = primals_280 = None
        getitem_197 = _native_batch_norm_legit_functional_39[0]
        getitem_198 = _native_batch_norm_legit_functional_39[1]
        getitem_199 = _native_batch_norm_legit_functional_39[2]
        getitem_200 = _native_batch_norm_legit_functional_39[3]
        getitem_201 = _native_batch_norm_legit_functional_39[4];  _native_batch_norm_legit_functional_39 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_197, relu_33);  getitem_197 = None
        relu_36 = torch.ops.aten.relu.default(add_51);  add_51 = None
        convolution_40 = torch.ops.aten.convolution.default(relu_36, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_52 = torch.ops.aten.add.Tensor(primals_284, 1);  primals_284 = None
        _native_batch_norm_legit_functional_40 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_40, primals_122, primals_123, primals_282, primals_283, True, 0.1, 1e-05);  primals_123 = primals_282 = primals_283 = None
        getitem_202 = _native_batch_norm_legit_functional_40[0]
        getitem_203 = _native_batch_norm_legit_functional_40[1]
        getitem_204 = _native_batch_norm_legit_functional_40[2]
        getitem_205 = _native_batch_norm_legit_functional_40[3]
        getitem_206 = _native_batch_norm_legit_functional_40[4];  _native_batch_norm_legit_functional_40 = None
        relu_37 = torch.ops.aten.relu.default(getitem_202);  getitem_202 = None
        convolution_41 = torch.ops.aten.convolution.default(relu_37, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_53 = torch.ops.aten.add.Tensor(primals_287, 1);  primals_287 = None
        _native_batch_norm_legit_functional_41 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_41, primals_125, primals_126, primals_285, primals_286, True, 0.1, 1e-05);  primals_126 = primals_285 = primals_286 = None
        getitem_207 = _native_batch_norm_legit_functional_41[0]
        getitem_208 = _native_batch_norm_legit_functional_41[1]
        getitem_209 = _native_batch_norm_legit_functional_41[2]
        getitem_210 = _native_batch_norm_legit_functional_41[3]
        getitem_211 = _native_batch_norm_legit_functional_41[4];  _native_batch_norm_legit_functional_41 = None
        relu_38 = torch.ops.aten.relu.default(getitem_207);  getitem_207 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_38, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_54 = torch.ops.aten.add.Tensor(primals_290, 1);  primals_290 = None
        _native_batch_norm_legit_functional_42 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_42, primals_128, primals_129, primals_288, primals_289, True, 0.1, 1e-05);  primals_129 = primals_288 = primals_289 = None
        getitem_212 = _native_batch_norm_legit_functional_42[0]
        getitem_213 = _native_batch_norm_legit_functional_42[1]
        getitem_214 = _native_batch_norm_legit_functional_42[2]
        getitem_215 = _native_batch_norm_legit_functional_42[3]
        getitem_216 = _native_batch_norm_legit_functional_42[4];  _native_batch_norm_legit_functional_42 = None
        add_55 = torch.ops.aten.add.Tensor(getitem_212, relu_36);  getitem_212 = None
        relu_39 = torch.ops.aten.relu.default(add_55);  add_55 = None
        convolution_43 = torch.ops.aten.convolution.default(relu_39, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_56 = torch.ops.aten.add.Tensor(primals_293, 1);  primals_293 = None
        _native_batch_norm_legit_functional_43 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_43, primals_131, primals_132, primals_291, primals_292, True, 0.1, 1e-05);  primals_132 = primals_291 = primals_292 = None
        getitem_217 = _native_batch_norm_legit_functional_43[0]
        getitem_218 = _native_batch_norm_legit_functional_43[1]
        getitem_219 = _native_batch_norm_legit_functional_43[2]
        getitem_220 = _native_batch_norm_legit_functional_43[3]
        getitem_221 = _native_batch_norm_legit_functional_43[4];  _native_batch_norm_legit_functional_43 = None
        relu_40 = torch.ops.aten.relu.default(getitem_217);  getitem_217 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_40, primals_133, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
        add_57 = torch.ops.aten.add.Tensor(primals_296, 1);  primals_296 = None
        _native_batch_norm_legit_functional_44 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_44, primals_134, primals_135, primals_294, primals_295, True, 0.1, 1e-05);  primals_135 = primals_294 = primals_295 = None
        getitem_222 = _native_batch_norm_legit_functional_44[0]
        getitem_223 = _native_batch_norm_legit_functional_44[1]
        getitem_224 = _native_batch_norm_legit_functional_44[2]
        getitem_225 = _native_batch_norm_legit_functional_44[3]
        getitem_226 = _native_batch_norm_legit_functional_44[4];  _native_batch_norm_legit_functional_44 = None
        relu_41 = torch.ops.aten.relu.default(getitem_222);  getitem_222 = None
        convolution_45 = torch.ops.aten.convolution.default(relu_41, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_58 = torch.ops.aten.add.Tensor(primals_299, 1);  primals_299 = None
        _native_batch_norm_legit_functional_45 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_45, primals_137, primals_138, primals_297, primals_298, True, 0.1, 1e-05);  primals_138 = primals_297 = primals_298 = None
        getitem_227 = _native_batch_norm_legit_functional_45[0]
        getitem_228 = _native_batch_norm_legit_functional_45[1]
        getitem_229 = _native_batch_norm_legit_functional_45[2]
        getitem_230 = _native_batch_norm_legit_functional_45[3]
        getitem_231 = _native_batch_norm_legit_functional_45[4];  _native_batch_norm_legit_functional_45 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_39, primals_139, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_59 = torch.ops.aten.add.Tensor(primals_302, 1);  primals_302 = None
        _native_batch_norm_legit_functional_46 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_46, primals_140, primals_141, primals_300, primals_301, True, 0.1, 1e-05);  primals_141 = primals_300 = primals_301 = None
        getitem_232 = _native_batch_norm_legit_functional_46[0]
        getitem_233 = _native_batch_norm_legit_functional_46[1]
        getitem_234 = _native_batch_norm_legit_functional_46[2]
        getitem_235 = _native_batch_norm_legit_functional_46[3]
        getitem_236 = _native_batch_norm_legit_functional_46[4];  _native_batch_norm_legit_functional_46 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_227, getitem_232);  getitem_227 = getitem_232 = None
        relu_42 = torch.ops.aten.relu.default(add_60);  add_60 = None
        convolution_47 = torch.ops.aten.convolution.default(relu_42, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_61 = torch.ops.aten.add.Tensor(primals_305, 1);  primals_305 = None
        _native_batch_norm_legit_functional_47 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_47, primals_143, primals_144, primals_303, primals_304, True, 0.1, 1e-05);  primals_144 = primals_303 = primals_304 = None
        getitem_237 = _native_batch_norm_legit_functional_47[0]
        getitem_238 = _native_batch_norm_legit_functional_47[1]
        getitem_239 = _native_batch_norm_legit_functional_47[2]
        getitem_240 = _native_batch_norm_legit_functional_47[3]
        getitem_241 = _native_batch_norm_legit_functional_47[4];  _native_batch_norm_legit_functional_47 = None
        relu_43 = torch.ops.aten.relu.default(getitem_237);  getitem_237 = None
        convolution_48 = torch.ops.aten.convolution.default(relu_43, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_62 = torch.ops.aten.add.Tensor(primals_308, 1);  primals_308 = None
        _native_batch_norm_legit_functional_48 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_48, primals_146, primals_147, primals_306, primals_307, True, 0.1, 1e-05);  primals_147 = primals_306 = primals_307 = None
        getitem_242 = _native_batch_norm_legit_functional_48[0]
        getitem_243 = _native_batch_norm_legit_functional_48[1]
        getitem_244 = _native_batch_norm_legit_functional_48[2]
        getitem_245 = _native_batch_norm_legit_functional_48[3]
        getitem_246 = _native_batch_norm_legit_functional_48[4];  _native_batch_norm_legit_functional_48 = None
        relu_44 = torch.ops.aten.relu.default(getitem_242);  getitem_242 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_44, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_63 = torch.ops.aten.add.Tensor(primals_311, 1);  primals_311 = None
        _native_batch_norm_legit_functional_49 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_49, primals_149, primals_150, primals_309, primals_310, True, 0.1, 1e-05);  primals_150 = primals_309 = primals_310 = None
        getitem_247 = _native_batch_norm_legit_functional_49[0]
        getitem_248 = _native_batch_norm_legit_functional_49[1]
        getitem_249 = _native_batch_norm_legit_functional_49[2]
        getitem_250 = _native_batch_norm_legit_functional_49[3]
        getitem_251 = _native_batch_norm_legit_functional_49[4];  _native_batch_norm_legit_functional_49 = None
        add_64 = torch.ops.aten.add.Tensor(getitem_247, relu_42);  getitem_247 = None
        relu_45 = torch.ops.aten.relu.default(add_64);  add_64 = None
        convolution_50 = torch.ops.aten.convolution.default(relu_45, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_65 = torch.ops.aten.add.Tensor(primals_314, 1);  primals_314 = None
        _native_batch_norm_legit_functional_50 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_50, primals_152, primals_153, primals_312, primals_313, True, 0.1, 1e-05);  primals_153 = primals_312 = primals_313 = None
        getitem_252 = _native_batch_norm_legit_functional_50[0]
        getitem_253 = _native_batch_norm_legit_functional_50[1]
        getitem_254 = _native_batch_norm_legit_functional_50[2]
        getitem_255 = _native_batch_norm_legit_functional_50[3]
        getitem_256 = _native_batch_norm_legit_functional_50[4];  _native_batch_norm_legit_functional_50 = None
        relu_46 = torch.ops.aten.relu.default(getitem_252);  getitem_252 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_46, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_66 = torch.ops.aten.add.Tensor(primals_317, 1);  primals_317 = None
        _native_batch_norm_legit_functional_51 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_51, primals_155, primals_156, primals_315, primals_316, True, 0.1, 1e-05);  primals_156 = primals_315 = primals_316 = None
        getitem_257 = _native_batch_norm_legit_functional_51[0]
        getitem_258 = _native_batch_norm_legit_functional_51[1]
        getitem_259 = _native_batch_norm_legit_functional_51[2]
        getitem_260 = _native_batch_norm_legit_functional_51[3]
        getitem_261 = _native_batch_norm_legit_functional_51[4];  _native_batch_norm_legit_functional_51 = None
        relu_47 = torch.ops.aten.relu.default(getitem_257);  getitem_257 = None
        convolution_52 = torch.ops.aten.convolution.default(relu_47, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_67 = torch.ops.aten.add.Tensor(primals_320, 1);  primals_320 = None
        _native_batch_norm_legit_functional_52 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_52, primals_158, primals_159, primals_318, primals_319, True, 0.1, 1e-05);  primals_159 = primals_318 = primals_319 = None
        getitem_262 = _native_batch_norm_legit_functional_52[0]
        getitem_263 = _native_batch_norm_legit_functional_52[1]
        getitem_264 = _native_batch_norm_legit_functional_52[2]
        getitem_265 = _native_batch_norm_legit_functional_52[3]
        getitem_266 = _native_batch_norm_legit_functional_52[4];  _native_batch_norm_legit_functional_52 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_262, relu_45);  getitem_262 = None
        relu_48 = torch.ops.aten.relu.default(add_68);  add_68 = None
        mean = torch.ops.aten.mean.dim(relu_48, [-1, -2], True)
        view = torch.ops.aten.view.default(mean, [1, 2048]);  mean = None
        t = torch.ops.aten.t.default(primals_160);  primals_160 = None
        addmm = torch.ops.aten.addmm.default(primals_161, view, t);  primals_161 = None
        return [getitem_3, getitem_4, add, getitem_10, getitem_11, add_1, getitem_15, getitem_16, add_2, getitem_20, getitem_21, add_3, getitem_25, getitem_26, add_4, getitem_30, getitem_31, add_6, getitem_35, getitem_36, add_7, getitem_40, getitem_41, add_8, getitem_45, getitem_46, add_10, getitem_50, getitem_51, add_11, getitem_55, getitem_56, add_12, getitem_60, getitem_61, add_14, getitem_65, getitem_66, add_15, getitem_70, getitem_71, add_16, getitem_75, getitem_76, add_17, getitem_80, getitem_81, add_19, getitem_85, getitem_86, add_20, getitem_90, getitem_91, add_21, getitem_95, getitem_96, add_23, getitem_100, getitem_101, add_24, getitem_105, getitem_106, add_25, getitem_110, getitem_111, add_27, getitem_115, getitem_116, add_28, getitem_120, getitem_121, add_29, getitem_125, getitem_126, add_31, getitem_130, getitem_131, add_32, getitem_135, getitem_136, add_33, getitem_140, getitem_141, add_34, getitem_145, getitem_146, add_36, getitem_150, getitem_151, add_37, getitem_155, getitem_156, add_38, getitem_160, getitem_161, add_40, getitem_165, getitem_166, add_41, getitem_170, getitem_171, add_42, getitem_175, getitem_176, add_44, getitem_180, getitem_181, add_45, getitem_185, getitem_186, add_46, getitem_190, getitem_191, add_48, getitem_195, getitem_196, add_49, getitem_200, getitem_201, add_50, getitem_205, getitem_206, add_52, getitem_210, getitem_211, add_53, getitem_215, getitem_216, add_54, getitem_220, getitem_221, add_56, getitem_225, getitem_226, add_57, getitem_230, getitem_231, add_58, getitem_235, getitem_236, add_59, getitem_240, getitem_241, add_61, getitem_245, getitem_246, add_62, getitem_250, getitem_251, add_63, getitem_255, getitem_256, add_65, getitem_260, getitem_261, add_66, getitem_265, getitem_266, add_67, addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, getitem_1, getitem_2, getitem_3, getitem_4, relu, getitem_5, getitem_6, convolution_1, getitem_8, getitem_9, getitem_10, getitem_11, relu_1, convolution_2, getitem_13, getitem_14, getitem_15, getitem_16, relu_2, convolution_3, getitem_18, getitem_19, getitem_20, getitem_21, convolution_4, getitem_23, getitem_24, getitem_25, getitem_26, relu_3, convolution_5, getitem_28, getitem_29, getitem_30, getitem_31, relu_4, convolution_6, getitem_33, getitem_34, getitem_35, getitem_36, relu_5, convolution_7, getitem_38, getitem_39, getitem_40, getitem_41, relu_6, convolution_8, getitem_43, getitem_44, getitem_45, getitem_46, relu_7, convolution_9, getitem_48, getitem_49, getitem_50, getitem_51, relu_8, convolution_10, getitem_53, getitem_54, getitem_55, getitem_56, relu_9, convolution_11, getitem_58, getitem_59, getitem_60, getitem_61, relu_10, convolution_12, getitem_63, getitem_64, getitem_65, getitem_66, relu_11, convolution_13, getitem_68, getitem_69, getitem_70, getitem_71, convolution_14, getitem_73, getitem_74, getitem_75, getitem_76, relu_12, convolution_15, getitem_78, getitem_79, getitem_80, getitem_81, relu_13, convolution_16, getitem_83, getitem_84, getitem_85, getitem_86, relu_14, convolution_17, getitem_88, getitem_89, getitem_90, getitem_91, relu_15, convolution_18, getitem_93, getitem_94, getitem_95, getitem_96, relu_16, convolution_19, getitem_98, getitem_99, getitem_100, getitem_101, relu_17, convolution_20, getitem_103, getitem_104, getitem_105, getitem_106, relu_18, convolution_21, getitem_108, getitem_109, getitem_110, getitem_111, relu_19, convolution_22, getitem_113, getitem_114, getitem_115, getitem_116, relu_20, convolution_23, getitem_118, getitem_119, getitem_120, getitem_121, relu_21, convolution_24, getitem_123, getitem_124, getitem_125, getitem_126, relu_22, convolution_25, getitem_128, getitem_129, getitem_130, getitem_131, relu_23, convolution_26, getitem_133, getitem_134, getitem_135, getitem_136, convolution_27, getitem_138, getitem_139, getitem_140, getitem_141, relu_24, convolution_28, getitem_143, getitem_144, getitem_145, getitem_146, relu_25, convolution_29, getitem_148, getitem_149, getitem_150, getitem_151, relu_26, convolution_30, getitem_153, getitem_154, getitem_155, getitem_156, relu_27, convolution_31, getitem_158, getitem_159, getitem_160, getitem_161, relu_28, convolution_32, getitem_163, getitem_164, getitem_165, getitem_166, relu_29, convolution_33, getitem_168, getitem_169, getitem_170, getitem_171, relu_30, convolution_34, getitem_173, getitem_174, getitem_175, getitem_176, relu_31, convolution_35, getitem_178, getitem_179, getitem_180, getitem_181, relu_32, convolution_36, getitem_183, getitem_184, getitem_185, getitem_186, relu_33, convolution_37, getitem_188, getitem_189, getitem_190, getitem_191, relu_34, convolution_38, getitem_193, getitem_194, getitem_195, getitem_196, relu_35, convolution_39, getitem_198, getitem_199, getitem_200, getitem_201, relu_36, convolution_40, getitem_203, getitem_204, getitem_205, getitem_206, relu_37, convolution_41, getitem_208, getitem_209, getitem_210, getitem_211, relu_38, convolution_42, getitem_213, getitem_214, getitem_215, getitem_216, relu_39, convolution_43, getitem_218, getitem_219, getitem_220, getitem_221, relu_40, convolution_44, getitem_223, getitem_224, getitem_225, getitem_226, relu_41, convolution_45, getitem_228, getitem_229, getitem_230, getitem_231, convolution_46, getitem_233, getitem_234, getitem_235, getitem_236, relu_42, convolution_47, getitem_238, getitem_239, getitem_240, getitem_241, relu_43, convolution_48, getitem_243, getitem_244, getitem_245, getitem_246, relu_44, convolution_49, getitem_248, getitem_249, getitem_250, getitem_251, relu_45, convolution_50, getitem_253, getitem_254, getitem_255, getitem_256, relu_46, convolution_51, getitem_258, getitem_259, getitem_260, getitem_261, relu_47, convolution_52, getitem_263, getitem_264, getitem_265, getitem_266, relu_48, view, t]
        
