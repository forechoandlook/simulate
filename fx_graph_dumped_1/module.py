
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
        self.load_state_dict(torch.load(r'fx_graph_dumped_1/state_dict.pt'))

    
    
    def forward(self, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_321, convolution, getitem_1, getitem_2, getitem_3, getitem_4, relu, getitem_5, getitem_6, convolution_1, getitem_8, getitem_9, getitem_10, getitem_11, relu_1, convolution_2, getitem_13, getitem_14, getitem_15, getitem_16, relu_2, convolution_3, getitem_18, getitem_19, getitem_20, getitem_21, convolution_4, getitem_23, getitem_24, getitem_25, getitem_26, relu_3, convolution_5, getitem_28, getitem_29, getitem_30, getitem_31, relu_4, convolution_6, getitem_33, getitem_34, getitem_35, getitem_36, relu_5, convolution_7, getitem_38, getitem_39, getitem_40, getitem_41, relu_6, convolution_8, getitem_43, getitem_44, getitem_45, getitem_46, relu_7, convolution_9, getitem_48, getitem_49, getitem_50, getitem_51, relu_8, convolution_10, getitem_53, getitem_54, getitem_55, getitem_56, relu_9, convolution_11, getitem_58, getitem_59, getitem_60, getitem_61, relu_10, convolution_12, getitem_63, getitem_64, getitem_65, getitem_66, relu_11, convolution_13, getitem_68, getitem_69, getitem_70, getitem_71, convolution_14, getitem_73, getitem_74, getitem_75, getitem_76, relu_12, convolution_15, getitem_78, getitem_79, getitem_80, getitem_81, relu_13, convolution_16, getitem_83, getitem_84, getitem_85, getitem_86, relu_14, convolution_17, getitem_88, getitem_89, getitem_90, getitem_91, relu_15, convolution_18, getitem_93, getitem_94, getitem_95, getitem_96, relu_16, convolution_19, getitem_98, getitem_99, getitem_100, getitem_101, relu_17, convolution_20, getitem_103, getitem_104, getitem_105, getitem_106, relu_18, convolution_21, getitem_108, getitem_109, getitem_110, getitem_111, relu_19, convolution_22, getitem_113, getitem_114, getitem_115, getitem_116, relu_20, convolution_23, getitem_118, getitem_119, getitem_120, getitem_121, relu_21, convolution_24, getitem_123, getitem_124, getitem_125, getitem_126, relu_22, convolution_25, getitem_128, getitem_129, getitem_130, getitem_131, relu_23, convolution_26, getitem_133, getitem_134, getitem_135, getitem_136, convolution_27, getitem_138, getitem_139, getitem_140, getitem_141, relu_24, convolution_28, getitem_143, getitem_144, getitem_145, getitem_146, relu_25, convolution_29, getitem_148, getitem_149, getitem_150, getitem_151, relu_26, convolution_30, getitem_153, getitem_154, getitem_155, getitem_156, relu_27, convolution_31, getitem_158, getitem_159, getitem_160, getitem_161, relu_28, convolution_32, getitem_163, getitem_164, getitem_165, getitem_166, relu_29, convolution_33, getitem_168, getitem_169, getitem_170, getitem_171, relu_30, convolution_34, getitem_173, getitem_174, getitem_175, getitem_176, relu_31, convolution_35, getitem_178, getitem_179, getitem_180, getitem_181, relu_32, convolution_36, getitem_183, getitem_184, getitem_185, getitem_186, relu_33, convolution_37, getitem_188, getitem_189, getitem_190, getitem_191, relu_34, convolution_38, getitem_193, getitem_194, getitem_195, getitem_196, relu_35, convolution_39, getitem_198, getitem_199, getitem_200, getitem_201, relu_36, convolution_40, getitem_203, getitem_204, getitem_205, getitem_206, relu_37, convolution_41, getitem_208, getitem_209, getitem_210, getitem_211, relu_38, convolution_42, getitem_213, getitem_214, getitem_215, getitem_216, relu_39, convolution_43, getitem_218, getitem_219, getitem_220, getitem_221, relu_40, convolution_44, getitem_223, getitem_224, getitem_225, getitem_226, relu_41, convolution_45, getitem_228, getitem_229, getitem_230, getitem_231, convolution_46, getitem_233, getitem_234, getitem_235, getitem_236, relu_42, convolution_47, getitem_238, getitem_239, getitem_240, getitem_241, relu_43, convolution_48, getitem_243, getitem_244, getitem_245, getitem_246, relu_44, convolution_49, getitem_248, getitem_249, getitem_250, getitem_251, relu_45, convolution_50, getitem_253, getitem_254, getitem_255, getitem_256, relu_46, convolution_51, getitem_258, getitem_259, getitem_260, getitem_261, relu_47, convolution_52, getitem_263, getitem_264, getitem_265, getitem_266, relu_48, view, t, tangents_1):
        t_1 = torch.ops.aten.t.default(t);  t = None
        mm = torch.ops.aten.mm.default(tangents_1, t_1);  t_1 = None
        t_2 = torch.ops.aten.t.default(tangents_1)
        mm_1 = torch.ops.aten.mm.default(t_2, view);  t_2 = view = None
        t_3 = torch.ops.aten.t.default(mm_1);  mm_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1 = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
        _to_copy = torch.ops.aten._to_copy.default(mm, dtype = torch.float16);  mm = None
        t_4 = torch.ops.aten.t.default(t_3);  t_3 = None
        view_2 = torch.ops.aten.view.default(_to_copy, [1, 2048, 1, 1]);  _to_copy = None
        expand = torch.ops.aten.expand.default(view_2, [1, 2048, 7, 7]);  view_2 = None
        div = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        detach_100 = torch.ops.aten.detach.default(relu_48);  relu_48 = None
        detach_101 = torch.ops.aten.detach.default(detach_100);  detach_100 = None
        threshold_backward = torch.ops.aten.threshold_backward.default(div, detach_101, 0);  div = detach_101 = None
        native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(threshold_backward, convolution_52, primals_158, getitem_265, getitem_266, getitem_263, getitem_264, True, 1e-05, [True, True, True]);  convolution_52 = primals_158 = getitem_265 = getitem_266 = getitem_263 = getitem_264 = None
        getitem_267 = native_batch_norm_backward[0]
        getitem_268 = native_batch_norm_backward[1]
        getitem_269 = native_batch_norm_backward[2];  native_batch_norm_backward = None
        convolution_backward = torch.ops.aten.convolution_backward.default(getitem_267, relu_47, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_267 = primals_157 = None
        getitem_270 = convolution_backward[0]
        getitem_271 = convolution_backward[1];  convolution_backward = None
        _to_copy_1 = torch.ops.aten._to_copy.default(getitem_271, dtype = torch.float32);  getitem_271 = None
        detach_104 = torch.ops.aten.detach.default(relu_47);  relu_47 = None
        detach_105 = torch.ops.aten.detach.default(detach_104);  detach_104 = None
        threshold_backward_1 = torch.ops.aten.threshold_backward.default(getitem_270, detach_105, 0);  getitem_270 = detach_105 = None
        native_batch_norm_backward_1 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_1, convolution_51, primals_155, getitem_260, getitem_261, getitem_258, getitem_259, True, 1e-05, [True, True, True]);  threshold_backward_1 = convolution_51 = primals_155 = getitem_260 = getitem_261 = getitem_258 = getitem_259 = None
        getitem_273 = native_batch_norm_backward_1[0]
        getitem_274 = native_batch_norm_backward_1[1]
        getitem_275 = native_batch_norm_backward_1[2];  native_batch_norm_backward_1 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_273, relu_46, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_273 = primals_154 = None
        getitem_276 = convolution_backward_1[0]
        getitem_277 = convolution_backward_1[1];  convolution_backward_1 = None
        _to_copy_2 = torch.ops.aten._to_copy.default(getitem_277, dtype = torch.float32);  getitem_277 = None
        detach_108 = torch.ops.aten.detach.default(relu_46);  relu_46 = None
        detach_109 = torch.ops.aten.detach.default(detach_108);  detach_108 = None
        threshold_backward_2 = torch.ops.aten.threshold_backward.default(getitem_276, detach_109, 0);  getitem_276 = detach_109 = None
        native_batch_norm_backward_2 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_2, convolution_50, primals_152, getitem_255, getitem_256, getitem_253, getitem_254, True, 1e-05, [True, True, True]);  threshold_backward_2 = convolution_50 = primals_152 = getitem_255 = getitem_256 = getitem_253 = getitem_254 = None
        getitem_279 = native_batch_norm_backward_2[0]
        getitem_280 = native_batch_norm_backward_2[1]
        getitem_281 = native_batch_norm_backward_2[2];  native_batch_norm_backward_2 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(getitem_279, relu_45, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_279 = primals_151 = None
        getitem_282 = convolution_backward_2[0]
        getitem_283 = convolution_backward_2[1];  convolution_backward_2 = None
        _to_copy_3 = torch.ops.aten._to_copy.default(getitem_283, dtype = torch.float32);  getitem_283 = None
        add_69 = torch.ops.aten.add.Tensor(threshold_backward, getitem_282);  threshold_backward = getitem_282 = None
        detach_112 = torch.ops.aten.detach.default(relu_45);  relu_45 = None
        detach_113 = torch.ops.aten.detach.default(detach_112);  detach_112 = None
        threshold_backward_3 = torch.ops.aten.threshold_backward.default(add_69, detach_113, 0);  add_69 = detach_113 = None
        native_batch_norm_backward_3 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_3, convolution_49, primals_149, getitem_250, getitem_251, getitem_248, getitem_249, True, 1e-05, [True, True, True]);  convolution_49 = primals_149 = getitem_250 = getitem_251 = getitem_248 = getitem_249 = None
        getitem_285 = native_batch_norm_backward_3[0]
        getitem_286 = native_batch_norm_backward_3[1]
        getitem_287 = native_batch_norm_backward_3[2];  native_batch_norm_backward_3 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_285, relu_44, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_285 = primals_148 = None
        getitem_288 = convolution_backward_3[0]
        getitem_289 = convolution_backward_3[1];  convolution_backward_3 = None
        _to_copy_4 = torch.ops.aten._to_copy.default(getitem_289, dtype = torch.float32);  getitem_289 = None
        detach_116 = torch.ops.aten.detach.default(relu_44);  relu_44 = None
        detach_117 = torch.ops.aten.detach.default(detach_116);  detach_116 = None
        threshold_backward_4 = torch.ops.aten.threshold_backward.default(getitem_288, detach_117, 0);  getitem_288 = detach_117 = None
        native_batch_norm_backward_4 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_4, convolution_48, primals_146, getitem_245, getitem_246, getitem_243, getitem_244, True, 1e-05, [True, True, True]);  threshold_backward_4 = convolution_48 = primals_146 = getitem_245 = getitem_246 = getitem_243 = getitem_244 = None
        getitem_291 = native_batch_norm_backward_4[0]
        getitem_292 = native_batch_norm_backward_4[1]
        getitem_293 = native_batch_norm_backward_4[2];  native_batch_norm_backward_4 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(getitem_291, relu_43, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_291 = primals_145 = None
        getitem_294 = convolution_backward_4[0]
        getitem_295 = convolution_backward_4[1];  convolution_backward_4 = None
        _to_copy_5 = torch.ops.aten._to_copy.default(getitem_295, dtype = torch.float32);  getitem_295 = None
        detach_120 = torch.ops.aten.detach.default(relu_43);  relu_43 = None
        detach_121 = torch.ops.aten.detach.default(detach_120);  detach_120 = None
        threshold_backward_5 = torch.ops.aten.threshold_backward.default(getitem_294, detach_121, 0);  getitem_294 = detach_121 = None
        native_batch_norm_backward_5 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_5, convolution_47, primals_143, getitem_240, getitem_241, getitem_238, getitem_239, True, 1e-05, [True, True, True]);  threshold_backward_5 = convolution_47 = primals_143 = getitem_240 = getitem_241 = getitem_238 = getitem_239 = None
        getitem_297 = native_batch_norm_backward_5[0]
        getitem_298 = native_batch_norm_backward_5[1]
        getitem_299 = native_batch_norm_backward_5[2];  native_batch_norm_backward_5 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_297, relu_42, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_297 = primals_142 = None
        getitem_300 = convolution_backward_5[0]
        getitem_301 = convolution_backward_5[1];  convolution_backward_5 = None
        _to_copy_6 = torch.ops.aten._to_copy.default(getitem_301, dtype = torch.float32);  getitem_301 = None
        add_70 = torch.ops.aten.add.Tensor(threshold_backward_3, getitem_300);  threshold_backward_3 = getitem_300 = None
        detach_124 = torch.ops.aten.detach.default(relu_42);  relu_42 = None
        detach_125 = torch.ops.aten.detach.default(detach_124);  detach_124 = None
        threshold_backward_6 = torch.ops.aten.threshold_backward.default(add_70, detach_125, 0);  add_70 = detach_125 = None
        native_batch_norm_backward_6 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_6, convolution_46, primals_140, getitem_235, getitem_236, getitem_233, getitem_234, True, 1e-05, [True, True, True]);  convolution_46 = primals_140 = getitem_235 = getitem_236 = getitem_233 = getitem_234 = None
        getitem_303 = native_batch_norm_backward_6[0]
        getitem_304 = native_batch_norm_backward_6[1]
        getitem_305 = native_batch_norm_backward_6[2];  native_batch_norm_backward_6 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(getitem_303, relu_39, primals_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_303 = primals_139 = None
        getitem_306 = convolution_backward_6[0]
        getitem_307 = convolution_backward_6[1];  convolution_backward_6 = None
        _to_copy_7 = torch.ops.aten._to_copy.default(getitem_307, dtype = torch.float32);  getitem_307 = None
        native_batch_norm_backward_7 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_6, convolution_45, primals_137, getitem_230, getitem_231, getitem_228, getitem_229, True, 1e-05, [True, True, True]);  threshold_backward_6 = convolution_45 = primals_137 = getitem_230 = getitem_231 = getitem_228 = getitem_229 = None
        getitem_309 = native_batch_norm_backward_7[0]
        getitem_310 = native_batch_norm_backward_7[1]
        getitem_311 = native_batch_norm_backward_7[2];  native_batch_norm_backward_7 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_309, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_309 = primals_136 = None
        getitem_312 = convolution_backward_7[0]
        getitem_313 = convolution_backward_7[1];  convolution_backward_7 = None
        _to_copy_8 = torch.ops.aten._to_copy.default(getitem_313, dtype = torch.float32);  getitem_313 = None
        detach_128 = torch.ops.aten.detach.default(relu_41);  relu_41 = None
        detach_129 = torch.ops.aten.detach.default(detach_128);  detach_128 = None
        threshold_backward_7 = torch.ops.aten.threshold_backward.default(getitem_312, detach_129, 0);  getitem_312 = detach_129 = None
        native_batch_norm_backward_8 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_7, convolution_44, primals_134, getitem_225, getitem_226, getitem_223, getitem_224, True, 1e-05, [True, True, True]);  threshold_backward_7 = convolution_44 = primals_134 = getitem_225 = getitem_226 = getitem_223 = getitem_224 = None
        getitem_315 = native_batch_norm_backward_8[0]
        getitem_316 = native_batch_norm_backward_8[1]
        getitem_317 = native_batch_norm_backward_8[2];  native_batch_norm_backward_8 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(getitem_315, relu_40, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_315 = primals_133 = None
        getitem_318 = convolution_backward_8[0]
        getitem_319 = convolution_backward_8[1];  convolution_backward_8 = None
        _to_copy_9 = torch.ops.aten._to_copy.default(getitem_319, dtype = torch.float32);  getitem_319 = None
        detach_132 = torch.ops.aten.detach.default(relu_40);  relu_40 = None
        detach_133 = torch.ops.aten.detach.default(detach_132);  detach_132 = None
        threshold_backward_8 = torch.ops.aten.threshold_backward.default(getitem_318, detach_133, 0);  getitem_318 = detach_133 = None
        native_batch_norm_backward_9 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_8, convolution_43, primals_131, getitem_220, getitem_221, getitem_218, getitem_219, True, 1e-05, [True, True, True]);  threshold_backward_8 = convolution_43 = primals_131 = getitem_220 = getitem_221 = getitem_218 = getitem_219 = None
        getitem_321 = native_batch_norm_backward_9[0]
        getitem_322 = native_batch_norm_backward_9[1]
        getitem_323 = native_batch_norm_backward_9[2];  native_batch_norm_backward_9 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(getitem_321, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_321 = primals_130 = None
        getitem_324 = convolution_backward_9[0]
        getitem_325 = convolution_backward_9[1];  convolution_backward_9 = None
        _to_copy_10 = torch.ops.aten._to_copy.default(getitem_325, dtype = torch.float32);  getitem_325 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_306, getitem_324);  getitem_306 = getitem_324 = None
        detach_136 = torch.ops.aten.detach.default(relu_39);  relu_39 = None
        detach_137 = torch.ops.aten.detach.default(detach_136);  detach_136 = None
        threshold_backward_9 = torch.ops.aten.threshold_backward.default(add_71, detach_137, 0);  add_71 = detach_137 = None
        native_batch_norm_backward_10 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_9, convolution_42, primals_128, getitem_215, getitem_216, getitem_213, getitem_214, True, 1e-05, [True, True, True]);  convolution_42 = primals_128 = getitem_215 = getitem_216 = getitem_213 = getitem_214 = None
        getitem_327 = native_batch_norm_backward_10[0]
        getitem_328 = native_batch_norm_backward_10[1]
        getitem_329 = native_batch_norm_backward_10[2];  native_batch_norm_backward_10 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(getitem_327, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_327 = primals_127 = None
        getitem_330 = convolution_backward_10[0]
        getitem_331 = convolution_backward_10[1];  convolution_backward_10 = None
        _to_copy_11 = torch.ops.aten._to_copy.default(getitem_331, dtype = torch.float32);  getitem_331 = None
        detach_140 = torch.ops.aten.detach.default(relu_38);  relu_38 = None
        detach_141 = torch.ops.aten.detach.default(detach_140);  detach_140 = None
        threshold_backward_10 = torch.ops.aten.threshold_backward.default(getitem_330, detach_141, 0);  getitem_330 = detach_141 = None
        native_batch_norm_backward_11 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_10, convolution_41, primals_125, getitem_210, getitem_211, getitem_208, getitem_209, True, 1e-05, [True, True, True]);  threshold_backward_10 = convolution_41 = primals_125 = getitem_210 = getitem_211 = getitem_208 = getitem_209 = None
        getitem_333 = native_batch_norm_backward_11[0]
        getitem_334 = native_batch_norm_backward_11[1]
        getitem_335 = native_batch_norm_backward_11[2];  native_batch_norm_backward_11 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(getitem_333, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_333 = primals_124 = None
        getitem_336 = convolution_backward_11[0]
        getitem_337 = convolution_backward_11[1];  convolution_backward_11 = None
        _to_copy_12 = torch.ops.aten._to_copy.default(getitem_337, dtype = torch.float32);  getitem_337 = None
        detach_144 = torch.ops.aten.detach.default(relu_37);  relu_37 = None
        detach_145 = torch.ops.aten.detach.default(detach_144);  detach_144 = None
        threshold_backward_11 = torch.ops.aten.threshold_backward.default(getitem_336, detach_145, 0);  getitem_336 = detach_145 = None
        native_batch_norm_backward_12 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_11, convolution_40, primals_122, getitem_205, getitem_206, getitem_203, getitem_204, True, 1e-05, [True, True, True]);  threshold_backward_11 = convolution_40 = primals_122 = getitem_205 = getitem_206 = getitem_203 = getitem_204 = None
        getitem_339 = native_batch_norm_backward_12[0]
        getitem_340 = native_batch_norm_backward_12[1]
        getitem_341 = native_batch_norm_backward_12[2];  native_batch_norm_backward_12 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(getitem_339, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_339 = primals_121 = None
        getitem_342 = convolution_backward_12[0]
        getitem_343 = convolution_backward_12[1];  convolution_backward_12 = None
        _to_copy_13 = torch.ops.aten._to_copy.default(getitem_343, dtype = torch.float32);  getitem_343 = None
        add_72 = torch.ops.aten.add.Tensor(threshold_backward_9, getitem_342);  threshold_backward_9 = getitem_342 = None
        detach_148 = torch.ops.aten.detach.default(relu_36);  relu_36 = None
        detach_149 = torch.ops.aten.detach.default(detach_148);  detach_148 = None
        threshold_backward_12 = torch.ops.aten.threshold_backward.default(add_72, detach_149, 0);  add_72 = detach_149 = None
        native_batch_norm_backward_13 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_12, convolution_39, primals_119, getitem_200, getitem_201, getitem_198, getitem_199, True, 1e-05, [True, True, True]);  convolution_39 = primals_119 = getitem_200 = getitem_201 = getitem_198 = getitem_199 = None
        getitem_345 = native_batch_norm_backward_13[0]
        getitem_346 = native_batch_norm_backward_13[1]
        getitem_347 = native_batch_norm_backward_13[2];  native_batch_norm_backward_13 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(getitem_345, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_345 = primals_118 = None
        getitem_348 = convolution_backward_13[0]
        getitem_349 = convolution_backward_13[1];  convolution_backward_13 = None
        _to_copy_14 = torch.ops.aten._to_copy.default(getitem_349, dtype = torch.float32);  getitem_349 = None
        detach_152 = torch.ops.aten.detach.default(relu_35);  relu_35 = None
        detach_153 = torch.ops.aten.detach.default(detach_152);  detach_152 = None
        threshold_backward_13 = torch.ops.aten.threshold_backward.default(getitem_348, detach_153, 0);  getitem_348 = detach_153 = None
        native_batch_norm_backward_14 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_13, convolution_38, primals_116, getitem_195, getitem_196, getitem_193, getitem_194, True, 1e-05, [True, True, True]);  threshold_backward_13 = convolution_38 = primals_116 = getitem_195 = getitem_196 = getitem_193 = getitem_194 = None
        getitem_351 = native_batch_norm_backward_14[0]
        getitem_352 = native_batch_norm_backward_14[1]
        getitem_353 = native_batch_norm_backward_14[2];  native_batch_norm_backward_14 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(getitem_351, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_351 = primals_115 = None
        getitem_354 = convolution_backward_14[0]
        getitem_355 = convolution_backward_14[1];  convolution_backward_14 = None
        _to_copy_15 = torch.ops.aten._to_copy.default(getitem_355, dtype = torch.float32);  getitem_355 = None
        detach_156 = torch.ops.aten.detach.default(relu_34);  relu_34 = None
        detach_157 = torch.ops.aten.detach.default(detach_156);  detach_156 = None
        threshold_backward_14 = torch.ops.aten.threshold_backward.default(getitem_354, detach_157, 0);  getitem_354 = detach_157 = None
        native_batch_norm_backward_15 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_14, convolution_37, primals_113, getitem_190, getitem_191, getitem_188, getitem_189, True, 1e-05, [True, True, True]);  threshold_backward_14 = convolution_37 = primals_113 = getitem_190 = getitem_191 = getitem_188 = getitem_189 = None
        getitem_357 = native_batch_norm_backward_15[0]
        getitem_358 = native_batch_norm_backward_15[1]
        getitem_359 = native_batch_norm_backward_15[2];  native_batch_norm_backward_15 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(getitem_357, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_357 = primals_112 = None
        getitem_360 = convolution_backward_15[0]
        getitem_361 = convolution_backward_15[1];  convolution_backward_15 = None
        _to_copy_16 = torch.ops.aten._to_copy.default(getitem_361, dtype = torch.float32);  getitem_361 = None
        add_73 = torch.ops.aten.add.Tensor(threshold_backward_12, getitem_360);  threshold_backward_12 = getitem_360 = None
        detach_160 = torch.ops.aten.detach.default(relu_33);  relu_33 = None
        detach_161 = torch.ops.aten.detach.default(detach_160);  detach_160 = None
        threshold_backward_15 = torch.ops.aten.threshold_backward.default(add_73, detach_161, 0);  add_73 = detach_161 = None
        native_batch_norm_backward_16 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_15, convolution_36, primals_110, getitem_185, getitem_186, getitem_183, getitem_184, True, 1e-05, [True, True, True]);  convolution_36 = primals_110 = getitem_185 = getitem_186 = getitem_183 = getitem_184 = None
        getitem_363 = native_batch_norm_backward_16[0]
        getitem_364 = native_batch_norm_backward_16[1]
        getitem_365 = native_batch_norm_backward_16[2];  native_batch_norm_backward_16 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(getitem_363, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_363 = primals_109 = None
        getitem_366 = convolution_backward_16[0]
        getitem_367 = convolution_backward_16[1];  convolution_backward_16 = None
        _to_copy_17 = torch.ops.aten._to_copy.default(getitem_367, dtype = torch.float32);  getitem_367 = None
        detach_164 = torch.ops.aten.detach.default(relu_32);  relu_32 = None
        detach_165 = torch.ops.aten.detach.default(detach_164);  detach_164 = None
        threshold_backward_16 = torch.ops.aten.threshold_backward.default(getitem_366, detach_165, 0);  getitem_366 = detach_165 = None
        native_batch_norm_backward_17 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_16, convolution_35, primals_107, getitem_180, getitem_181, getitem_178, getitem_179, True, 1e-05, [True, True, True]);  threshold_backward_16 = convolution_35 = primals_107 = getitem_180 = getitem_181 = getitem_178 = getitem_179 = None
        getitem_369 = native_batch_norm_backward_17[0]
        getitem_370 = native_batch_norm_backward_17[1]
        getitem_371 = native_batch_norm_backward_17[2];  native_batch_norm_backward_17 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_369, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_369 = primals_106 = None
        getitem_372 = convolution_backward_17[0]
        getitem_373 = convolution_backward_17[1];  convolution_backward_17 = None
        _to_copy_18 = torch.ops.aten._to_copy.default(getitem_373, dtype = torch.float32);  getitem_373 = None
        detach_168 = torch.ops.aten.detach.default(relu_31);  relu_31 = None
        detach_169 = torch.ops.aten.detach.default(detach_168);  detach_168 = None
        threshold_backward_17 = torch.ops.aten.threshold_backward.default(getitem_372, detach_169, 0);  getitem_372 = detach_169 = None
        native_batch_norm_backward_18 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_17, convolution_34, primals_104, getitem_175, getitem_176, getitem_173, getitem_174, True, 1e-05, [True, True, True]);  threshold_backward_17 = convolution_34 = primals_104 = getitem_175 = getitem_176 = getitem_173 = getitem_174 = None
        getitem_375 = native_batch_norm_backward_18[0]
        getitem_376 = native_batch_norm_backward_18[1]
        getitem_377 = native_batch_norm_backward_18[2];  native_batch_norm_backward_18 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(getitem_375, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_375 = primals_103 = None
        getitem_378 = convolution_backward_18[0]
        getitem_379 = convolution_backward_18[1];  convolution_backward_18 = None
        _to_copy_19 = torch.ops.aten._to_copy.default(getitem_379, dtype = torch.float32);  getitem_379 = None
        add_74 = torch.ops.aten.add.Tensor(threshold_backward_15, getitem_378);  threshold_backward_15 = getitem_378 = None
        detach_172 = torch.ops.aten.detach.default(relu_30);  relu_30 = None
        detach_173 = torch.ops.aten.detach.default(detach_172);  detach_172 = None
        threshold_backward_18 = torch.ops.aten.threshold_backward.default(add_74, detach_173, 0);  add_74 = detach_173 = None
        native_batch_norm_backward_19 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_18, convolution_33, primals_101, getitem_170, getitem_171, getitem_168, getitem_169, True, 1e-05, [True, True, True]);  convolution_33 = primals_101 = getitem_170 = getitem_171 = getitem_168 = getitem_169 = None
        getitem_381 = native_batch_norm_backward_19[0]
        getitem_382 = native_batch_norm_backward_19[1]
        getitem_383 = native_batch_norm_backward_19[2];  native_batch_norm_backward_19 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(getitem_381, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_381 = primals_100 = None
        getitem_384 = convolution_backward_19[0]
        getitem_385 = convolution_backward_19[1];  convolution_backward_19 = None
        _to_copy_20 = torch.ops.aten._to_copy.default(getitem_385, dtype = torch.float32);  getitem_385 = None
        detach_176 = torch.ops.aten.detach.default(relu_29);  relu_29 = None
        detach_177 = torch.ops.aten.detach.default(detach_176);  detach_176 = None
        threshold_backward_19 = torch.ops.aten.threshold_backward.default(getitem_384, detach_177, 0);  getitem_384 = detach_177 = None
        native_batch_norm_backward_20 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_19, convolution_32, primals_98, getitem_165, getitem_166, getitem_163, getitem_164, True, 1e-05, [True, True, True]);  threshold_backward_19 = convolution_32 = primals_98 = getitem_165 = getitem_166 = getitem_163 = getitem_164 = None
        getitem_387 = native_batch_norm_backward_20[0]
        getitem_388 = native_batch_norm_backward_20[1]
        getitem_389 = native_batch_norm_backward_20[2];  native_batch_norm_backward_20 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(getitem_387, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_387 = primals_97 = None
        getitem_390 = convolution_backward_20[0]
        getitem_391 = convolution_backward_20[1];  convolution_backward_20 = None
        _to_copy_21 = torch.ops.aten._to_copy.default(getitem_391, dtype = torch.float32);  getitem_391 = None
        detach_180 = torch.ops.aten.detach.default(relu_28);  relu_28 = None
        detach_181 = torch.ops.aten.detach.default(detach_180);  detach_180 = None
        threshold_backward_20 = torch.ops.aten.threshold_backward.default(getitem_390, detach_181, 0);  getitem_390 = detach_181 = None
        native_batch_norm_backward_21 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_20, convolution_31, primals_95, getitem_160, getitem_161, getitem_158, getitem_159, True, 1e-05, [True, True, True]);  threshold_backward_20 = convolution_31 = primals_95 = getitem_160 = getitem_161 = getitem_158 = getitem_159 = None
        getitem_393 = native_batch_norm_backward_21[0]
        getitem_394 = native_batch_norm_backward_21[1]
        getitem_395 = native_batch_norm_backward_21[2];  native_batch_norm_backward_21 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_393, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_393 = primals_94 = None
        getitem_396 = convolution_backward_21[0]
        getitem_397 = convolution_backward_21[1];  convolution_backward_21 = None
        _to_copy_22 = torch.ops.aten._to_copy.default(getitem_397, dtype = torch.float32);  getitem_397 = None
        add_75 = torch.ops.aten.add.Tensor(threshold_backward_18, getitem_396);  threshold_backward_18 = getitem_396 = None
        detach_184 = torch.ops.aten.detach.default(relu_27);  relu_27 = None
        detach_185 = torch.ops.aten.detach.default(detach_184);  detach_184 = None
        threshold_backward_21 = torch.ops.aten.threshold_backward.default(add_75, detach_185, 0);  add_75 = detach_185 = None
        native_batch_norm_backward_22 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_21, convolution_30, primals_92, getitem_155, getitem_156, getitem_153, getitem_154, True, 1e-05, [True, True, True]);  convolution_30 = primals_92 = getitem_155 = getitem_156 = getitem_153 = getitem_154 = None
        getitem_399 = native_batch_norm_backward_22[0]
        getitem_400 = native_batch_norm_backward_22[1]
        getitem_401 = native_batch_norm_backward_22[2];  native_batch_norm_backward_22 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(getitem_399, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_399 = primals_91 = None
        getitem_402 = convolution_backward_22[0]
        getitem_403 = convolution_backward_22[1];  convolution_backward_22 = None
        _to_copy_23 = torch.ops.aten._to_copy.default(getitem_403, dtype = torch.float32);  getitem_403 = None
        detach_188 = torch.ops.aten.detach.default(relu_26);  relu_26 = None
        detach_189 = torch.ops.aten.detach.default(detach_188);  detach_188 = None
        threshold_backward_22 = torch.ops.aten.threshold_backward.default(getitem_402, detach_189, 0);  getitem_402 = detach_189 = None
        native_batch_norm_backward_23 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_22, convolution_29, primals_89, getitem_150, getitem_151, getitem_148, getitem_149, True, 1e-05, [True, True, True]);  threshold_backward_22 = convolution_29 = primals_89 = getitem_150 = getitem_151 = getitem_148 = getitem_149 = None
        getitem_405 = native_batch_norm_backward_23[0]
        getitem_406 = native_batch_norm_backward_23[1]
        getitem_407 = native_batch_norm_backward_23[2];  native_batch_norm_backward_23 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_405, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_405 = primals_88 = None
        getitem_408 = convolution_backward_23[0]
        getitem_409 = convolution_backward_23[1];  convolution_backward_23 = None
        _to_copy_24 = torch.ops.aten._to_copy.default(getitem_409, dtype = torch.float32);  getitem_409 = None
        detach_192 = torch.ops.aten.detach.default(relu_25);  relu_25 = None
        detach_193 = torch.ops.aten.detach.default(detach_192);  detach_192 = None
        threshold_backward_23 = torch.ops.aten.threshold_backward.default(getitem_408, detach_193, 0);  getitem_408 = detach_193 = None
        native_batch_norm_backward_24 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_23, convolution_28, primals_86, getitem_145, getitem_146, getitem_143, getitem_144, True, 1e-05, [True, True, True]);  threshold_backward_23 = convolution_28 = primals_86 = getitem_145 = getitem_146 = getitem_143 = getitem_144 = None
        getitem_411 = native_batch_norm_backward_24[0]
        getitem_412 = native_batch_norm_backward_24[1]
        getitem_413 = native_batch_norm_backward_24[2];  native_batch_norm_backward_24 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(getitem_411, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_411 = primals_85 = None
        getitem_414 = convolution_backward_24[0]
        getitem_415 = convolution_backward_24[1];  convolution_backward_24 = None
        _to_copy_25 = torch.ops.aten._to_copy.default(getitem_415, dtype = torch.float32);  getitem_415 = None
        add_76 = torch.ops.aten.add.Tensor(threshold_backward_21, getitem_414);  threshold_backward_21 = getitem_414 = None
        detach_196 = torch.ops.aten.detach.default(relu_24);  relu_24 = None
        detach_197 = torch.ops.aten.detach.default(detach_196);  detach_196 = None
        threshold_backward_24 = torch.ops.aten.threshold_backward.default(add_76, detach_197, 0);  add_76 = detach_197 = None
        native_batch_norm_backward_25 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_24, convolution_27, primals_83, getitem_140, getitem_141, getitem_138, getitem_139, True, 1e-05, [True, True, True]);  convolution_27 = primals_83 = getitem_140 = getitem_141 = getitem_138 = getitem_139 = None
        getitem_417 = native_batch_norm_backward_25[0]
        getitem_418 = native_batch_norm_backward_25[1]
        getitem_419 = native_batch_norm_backward_25[2];  native_batch_norm_backward_25 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(getitem_417, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_417 = primals_82 = None
        getitem_420 = convolution_backward_25[0]
        getitem_421 = convolution_backward_25[1];  convolution_backward_25 = None
        _to_copy_26 = torch.ops.aten._to_copy.default(getitem_421, dtype = torch.float32);  getitem_421 = None
        native_batch_norm_backward_26 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_24, convolution_26, primals_80, getitem_135, getitem_136, getitem_133, getitem_134, True, 1e-05, [True, True, True]);  threshold_backward_24 = convolution_26 = primals_80 = getitem_135 = getitem_136 = getitem_133 = getitem_134 = None
        getitem_423 = native_batch_norm_backward_26[0]
        getitem_424 = native_batch_norm_backward_26[1]
        getitem_425 = native_batch_norm_backward_26[2];  native_batch_norm_backward_26 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(getitem_423, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_423 = primals_79 = None
        getitem_426 = convolution_backward_26[0]
        getitem_427 = convolution_backward_26[1];  convolution_backward_26 = None
        _to_copy_27 = torch.ops.aten._to_copy.default(getitem_427, dtype = torch.float32);  getitem_427 = None
        detach_200 = torch.ops.aten.detach.default(relu_23);  relu_23 = None
        detach_201 = torch.ops.aten.detach.default(detach_200);  detach_200 = None
        threshold_backward_25 = torch.ops.aten.threshold_backward.default(getitem_426, detach_201, 0);  getitem_426 = detach_201 = None
        native_batch_norm_backward_27 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_25, convolution_25, primals_77, getitem_130, getitem_131, getitem_128, getitem_129, True, 1e-05, [True, True, True]);  threshold_backward_25 = convolution_25 = primals_77 = getitem_130 = getitem_131 = getitem_128 = getitem_129 = None
        getitem_429 = native_batch_norm_backward_27[0]
        getitem_430 = native_batch_norm_backward_27[1]
        getitem_431 = native_batch_norm_backward_27[2];  native_batch_norm_backward_27 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(getitem_429, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_429 = primals_76 = None
        getitem_432 = convolution_backward_27[0]
        getitem_433 = convolution_backward_27[1];  convolution_backward_27 = None
        _to_copy_28 = torch.ops.aten._to_copy.default(getitem_433, dtype = torch.float32);  getitem_433 = None
        detach_204 = torch.ops.aten.detach.default(relu_22);  relu_22 = None
        detach_205 = torch.ops.aten.detach.default(detach_204);  detach_204 = None
        threshold_backward_26 = torch.ops.aten.threshold_backward.default(getitem_432, detach_205, 0);  getitem_432 = detach_205 = None
        native_batch_norm_backward_28 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_26, convolution_24, primals_74, getitem_125, getitem_126, getitem_123, getitem_124, True, 1e-05, [True, True, True]);  threshold_backward_26 = convolution_24 = primals_74 = getitem_125 = getitem_126 = getitem_123 = getitem_124 = None
        getitem_435 = native_batch_norm_backward_28[0]
        getitem_436 = native_batch_norm_backward_28[1]
        getitem_437 = native_batch_norm_backward_28[2];  native_batch_norm_backward_28 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(getitem_435, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_435 = primals_73 = None
        getitem_438 = convolution_backward_28[0]
        getitem_439 = convolution_backward_28[1];  convolution_backward_28 = None
        _to_copy_29 = torch.ops.aten._to_copy.default(getitem_439, dtype = torch.float32);  getitem_439 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_420, getitem_438);  getitem_420 = getitem_438 = None
        detach_208 = torch.ops.aten.detach.default(relu_21);  relu_21 = None
        detach_209 = torch.ops.aten.detach.default(detach_208);  detach_208 = None
        threshold_backward_27 = torch.ops.aten.threshold_backward.default(add_77, detach_209, 0);  add_77 = detach_209 = None
        native_batch_norm_backward_29 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_27, convolution_23, primals_71, getitem_120, getitem_121, getitem_118, getitem_119, True, 1e-05, [True, True, True]);  convolution_23 = primals_71 = getitem_120 = getitem_121 = getitem_118 = getitem_119 = None
        getitem_441 = native_batch_norm_backward_29[0]
        getitem_442 = native_batch_norm_backward_29[1]
        getitem_443 = native_batch_norm_backward_29[2];  native_batch_norm_backward_29 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(getitem_441, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_441 = primals_70 = None
        getitem_444 = convolution_backward_29[0]
        getitem_445 = convolution_backward_29[1];  convolution_backward_29 = None
        _to_copy_30 = torch.ops.aten._to_copy.default(getitem_445, dtype = torch.float32);  getitem_445 = None
        detach_212 = torch.ops.aten.detach.default(relu_20);  relu_20 = None
        detach_213 = torch.ops.aten.detach.default(detach_212);  detach_212 = None
        threshold_backward_28 = torch.ops.aten.threshold_backward.default(getitem_444, detach_213, 0);  getitem_444 = detach_213 = None
        native_batch_norm_backward_30 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_28, convolution_22, primals_68, getitem_115, getitem_116, getitem_113, getitem_114, True, 1e-05, [True, True, True]);  threshold_backward_28 = convolution_22 = primals_68 = getitem_115 = getitem_116 = getitem_113 = getitem_114 = None
        getitem_447 = native_batch_norm_backward_30[0]
        getitem_448 = native_batch_norm_backward_30[1]
        getitem_449 = native_batch_norm_backward_30[2];  native_batch_norm_backward_30 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(getitem_447, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_447 = primals_67 = None
        getitem_450 = convolution_backward_30[0]
        getitem_451 = convolution_backward_30[1];  convolution_backward_30 = None
        _to_copy_31 = torch.ops.aten._to_copy.default(getitem_451, dtype = torch.float32);  getitem_451 = None
        detach_216 = torch.ops.aten.detach.default(relu_19);  relu_19 = None
        detach_217 = torch.ops.aten.detach.default(detach_216);  detach_216 = None
        threshold_backward_29 = torch.ops.aten.threshold_backward.default(getitem_450, detach_217, 0);  getitem_450 = detach_217 = None
        native_batch_norm_backward_31 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_29, convolution_21, primals_65, getitem_110, getitem_111, getitem_108, getitem_109, True, 1e-05, [True, True, True]);  threshold_backward_29 = convolution_21 = primals_65 = getitem_110 = getitem_111 = getitem_108 = getitem_109 = None
        getitem_453 = native_batch_norm_backward_31[0]
        getitem_454 = native_batch_norm_backward_31[1]
        getitem_455 = native_batch_norm_backward_31[2];  native_batch_norm_backward_31 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(getitem_453, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_453 = primals_64 = None
        getitem_456 = convolution_backward_31[0]
        getitem_457 = convolution_backward_31[1];  convolution_backward_31 = None
        _to_copy_32 = torch.ops.aten._to_copy.default(getitem_457, dtype = torch.float32);  getitem_457 = None
        add_78 = torch.ops.aten.add.Tensor(threshold_backward_27, getitem_456);  threshold_backward_27 = getitem_456 = None
        detach_220 = torch.ops.aten.detach.default(relu_18);  relu_18 = None
        detach_221 = torch.ops.aten.detach.default(detach_220);  detach_220 = None
        threshold_backward_30 = torch.ops.aten.threshold_backward.default(add_78, detach_221, 0);  add_78 = detach_221 = None
        native_batch_norm_backward_32 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_30, convolution_20, primals_62, getitem_105, getitem_106, getitem_103, getitem_104, True, 1e-05, [True, True, True]);  convolution_20 = primals_62 = getitem_105 = getitem_106 = getitem_103 = getitem_104 = None
        getitem_459 = native_batch_norm_backward_32[0]
        getitem_460 = native_batch_norm_backward_32[1]
        getitem_461 = native_batch_norm_backward_32[2];  native_batch_norm_backward_32 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(getitem_459, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_459 = primals_61 = None
        getitem_462 = convolution_backward_32[0]
        getitem_463 = convolution_backward_32[1];  convolution_backward_32 = None
        _to_copy_33 = torch.ops.aten._to_copy.default(getitem_463, dtype = torch.float32);  getitem_463 = None
        detach_224 = torch.ops.aten.detach.default(relu_17);  relu_17 = None
        detach_225 = torch.ops.aten.detach.default(detach_224);  detach_224 = None
        threshold_backward_31 = torch.ops.aten.threshold_backward.default(getitem_462, detach_225, 0);  getitem_462 = detach_225 = None
        native_batch_norm_backward_33 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_31, convolution_19, primals_59, getitem_100, getitem_101, getitem_98, getitem_99, True, 1e-05, [True, True, True]);  threshold_backward_31 = convolution_19 = primals_59 = getitem_100 = getitem_101 = getitem_98 = getitem_99 = None
        getitem_465 = native_batch_norm_backward_33[0]
        getitem_466 = native_batch_norm_backward_33[1]
        getitem_467 = native_batch_norm_backward_33[2];  native_batch_norm_backward_33 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(getitem_465, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_465 = primals_58 = None
        getitem_468 = convolution_backward_33[0]
        getitem_469 = convolution_backward_33[1];  convolution_backward_33 = None
        _to_copy_34 = torch.ops.aten._to_copy.default(getitem_469, dtype = torch.float32);  getitem_469 = None
        detach_228 = torch.ops.aten.detach.default(relu_16);  relu_16 = None
        detach_229 = torch.ops.aten.detach.default(detach_228);  detach_228 = None
        threshold_backward_32 = torch.ops.aten.threshold_backward.default(getitem_468, detach_229, 0);  getitem_468 = detach_229 = None
        native_batch_norm_backward_34 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_32, convolution_18, primals_56, getitem_95, getitem_96, getitem_93, getitem_94, True, 1e-05, [True, True, True]);  threshold_backward_32 = convolution_18 = primals_56 = getitem_95 = getitem_96 = getitem_93 = getitem_94 = None
        getitem_471 = native_batch_norm_backward_34[0]
        getitem_472 = native_batch_norm_backward_34[1]
        getitem_473 = native_batch_norm_backward_34[2];  native_batch_norm_backward_34 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(getitem_471, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_471 = primals_55 = None
        getitem_474 = convolution_backward_34[0]
        getitem_475 = convolution_backward_34[1];  convolution_backward_34 = None
        _to_copy_35 = torch.ops.aten._to_copy.default(getitem_475, dtype = torch.float32);  getitem_475 = None
        add_79 = torch.ops.aten.add.Tensor(threshold_backward_30, getitem_474);  threshold_backward_30 = getitem_474 = None
        detach_232 = torch.ops.aten.detach.default(relu_15);  relu_15 = None
        detach_233 = torch.ops.aten.detach.default(detach_232);  detach_232 = None
        threshold_backward_33 = torch.ops.aten.threshold_backward.default(add_79, detach_233, 0);  add_79 = detach_233 = None
        native_batch_norm_backward_35 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_33, convolution_17, primals_53, getitem_90, getitem_91, getitem_88, getitem_89, True, 1e-05, [True, True, True]);  convolution_17 = primals_53 = getitem_90 = getitem_91 = getitem_88 = getitem_89 = None
        getitem_477 = native_batch_norm_backward_35[0]
        getitem_478 = native_batch_norm_backward_35[1]
        getitem_479 = native_batch_norm_backward_35[2];  native_batch_norm_backward_35 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(getitem_477, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_477 = primals_52 = None
        getitem_480 = convolution_backward_35[0]
        getitem_481 = convolution_backward_35[1];  convolution_backward_35 = None
        _to_copy_36 = torch.ops.aten._to_copy.default(getitem_481, dtype = torch.float32);  getitem_481 = None
        detach_236 = torch.ops.aten.detach.default(relu_14);  relu_14 = None
        detach_237 = torch.ops.aten.detach.default(detach_236);  detach_236 = None
        threshold_backward_34 = torch.ops.aten.threshold_backward.default(getitem_480, detach_237, 0);  getitem_480 = detach_237 = None
        native_batch_norm_backward_36 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_34, convolution_16, primals_50, getitem_85, getitem_86, getitem_83, getitem_84, True, 1e-05, [True, True, True]);  threshold_backward_34 = convolution_16 = primals_50 = getitem_85 = getitem_86 = getitem_83 = getitem_84 = None
        getitem_483 = native_batch_norm_backward_36[0]
        getitem_484 = native_batch_norm_backward_36[1]
        getitem_485 = native_batch_norm_backward_36[2];  native_batch_norm_backward_36 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(getitem_483, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_483 = primals_49 = None
        getitem_486 = convolution_backward_36[0]
        getitem_487 = convolution_backward_36[1];  convolution_backward_36 = None
        _to_copy_37 = torch.ops.aten._to_copy.default(getitem_487, dtype = torch.float32);  getitem_487 = None
        detach_240 = torch.ops.aten.detach.default(relu_13);  relu_13 = None
        detach_241 = torch.ops.aten.detach.default(detach_240);  detach_240 = None
        threshold_backward_35 = torch.ops.aten.threshold_backward.default(getitem_486, detach_241, 0);  getitem_486 = detach_241 = None
        native_batch_norm_backward_37 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_35, convolution_15, primals_47, getitem_80, getitem_81, getitem_78, getitem_79, True, 1e-05, [True, True, True]);  threshold_backward_35 = convolution_15 = primals_47 = getitem_80 = getitem_81 = getitem_78 = getitem_79 = None
        getitem_489 = native_batch_norm_backward_37[0]
        getitem_490 = native_batch_norm_backward_37[1]
        getitem_491 = native_batch_norm_backward_37[2];  native_batch_norm_backward_37 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(getitem_489, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_489 = primals_46 = None
        getitem_492 = convolution_backward_37[0]
        getitem_493 = convolution_backward_37[1];  convolution_backward_37 = None
        _to_copy_38 = torch.ops.aten._to_copy.default(getitem_493, dtype = torch.float32);  getitem_493 = None
        add_80 = torch.ops.aten.add.Tensor(threshold_backward_33, getitem_492);  threshold_backward_33 = getitem_492 = None
        detach_244 = torch.ops.aten.detach.default(relu_12);  relu_12 = None
        detach_245 = torch.ops.aten.detach.default(detach_244);  detach_244 = None
        threshold_backward_36 = torch.ops.aten.threshold_backward.default(add_80, detach_245, 0);  add_80 = detach_245 = None
        native_batch_norm_backward_38 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_36, convolution_14, primals_44, getitem_75, getitem_76, getitem_73, getitem_74, True, 1e-05, [True, True, True]);  convolution_14 = primals_44 = getitem_75 = getitem_76 = getitem_73 = getitem_74 = None
        getitem_495 = native_batch_norm_backward_38[0]
        getitem_496 = native_batch_norm_backward_38[1]
        getitem_497 = native_batch_norm_backward_38[2];  native_batch_norm_backward_38 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(getitem_495, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_495 = primals_43 = None
        getitem_498 = convolution_backward_38[0]
        getitem_499 = convolution_backward_38[1];  convolution_backward_38 = None
        _to_copy_39 = torch.ops.aten._to_copy.default(getitem_499, dtype = torch.float32);  getitem_499 = None
        native_batch_norm_backward_39 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_36, convolution_13, primals_41, getitem_70, getitem_71, getitem_68, getitem_69, True, 1e-05, [True, True, True]);  threshold_backward_36 = convolution_13 = primals_41 = getitem_70 = getitem_71 = getitem_68 = getitem_69 = None
        getitem_501 = native_batch_norm_backward_39[0]
        getitem_502 = native_batch_norm_backward_39[1]
        getitem_503 = native_batch_norm_backward_39[2];  native_batch_norm_backward_39 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(getitem_501, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_501 = primals_40 = None
        getitem_504 = convolution_backward_39[0]
        getitem_505 = convolution_backward_39[1];  convolution_backward_39 = None
        _to_copy_40 = torch.ops.aten._to_copy.default(getitem_505, dtype = torch.float32);  getitem_505 = None
        detach_248 = torch.ops.aten.detach.default(relu_11);  relu_11 = None
        detach_249 = torch.ops.aten.detach.default(detach_248);  detach_248 = None
        threshold_backward_37 = torch.ops.aten.threshold_backward.default(getitem_504, detach_249, 0);  getitem_504 = detach_249 = None
        native_batch_norm_backward_40 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_37, convolution_12, primals_38, getitem_65, getitem_66, getitem_63, getitem_64, True, 1e-05, [True, True, True]);  threshold_backward_37 = convolution_12 = primals_38 = getitem_65 = getitem_66 = getitem_63 = getitem_64 = None
        getitem_507 = native_batch_norm_backward_40[0]
        getitem_508 = native_batch_norm_backward_40[1]
        getitem_509 = native_batch_norm_backward_40[2];  native_batch_norm_backward_40 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(getitem_507, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_507 = primals_37 = None
        getitem_510 = convolution_backward_40[0]
        getitem_511 = convolution_backward_40[1];  convolution_backward_40 = None
        _to_copy_41 = torch.ops.aten._to_copy.default(getitem_511, dtype = torch.float32);  getitem_511 = None
        detach_252 = torch.ops.aten.detach.default(relu_10);  relu_10 = None
        detach_253 = torch.ops.aten.detach.default(detach_252);  detach_252 = None
        threshold_backward_38 = torch.ops.aten.threshold_backward.default(getitem_510, detach_253, 0);  getitem_510 = detach_253 = None
        native_batch_norm_backward_41 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_38, convolution_11, primals_35, getitem_60, getitem_61, getitem_58, getitem_59, True, 1e-05, [True, True, True]);  threshold_backward_38 = convolution_11 = primals_35 = getitem_60 = getitem_61 = getitem_58 = getitem_59 = None
        getitem_513 = native_batch_norm_backward_41[0]
        getitem_514 = native_batch_norm_backward_41[1]
        getitem_515 = native_batch_norm_backward_41[2];  native_batch_norm_backward_41 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(getitem_513, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_513 = primals_34 = None
        getitem_516 = convolution_backward_41[0]
        getitem_517 = convolution_backward_41[1];  convolution_backward_41 = None
        _to_copy_42 = torch.ops.aten._to_copy.default(getitem_517, dtype = torch.float32);  getitem_517 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_498, getitem_516);  getitem_498 = getitem_516 = None
        detach_256 = torch.ops.aten.detach.default(relu_9);  relu_9 = None
        detach_257 = torch.ops.aten.detach.default(detach_256);  detach_256 = None
        threshold_backward_39 = torch.ops.aten.threshold_backward.default(add_81, detach_257, 0);  add_81 = detach_257 = None
        native_batch_norm_backward_42 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_39, convolution_10, primals_32, getitem_55, getitem_56, getitem_53, getitem_54, True, 1e-05, [True, True, True]);  convolution_10 = primals_32 = getitem_55 = getitem_56 = getitem_53 = getitem_54 = None
        getitem_519 = native_batch_norm_backward_42[0]
        getitem_520 = native_batch_norm_backward_42[1]
        getitem_521 = native_batch_norm_backward_42[2];  native_batch_norm_backward_42 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(getitem_519, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_519 = primals_31 = None
        getitem_522 = convolution_backward_42[0]
        getitem_523 = convolution_backward_42[1];  convolution_backward_42 = None
        _to_copy_43 = torch.ops.aten._to_copy.default(getitem_523, dtype = torch.float32);  getitem_523 = None
        detach_260 = torch.ops.aten.detach.default(relu_8);  relu_8 = None
        detach_261 = torch.ops.aten.detach.default(detach_260);  detach_260 = None
        threshold_backward_40 = torch.ops.aten.threshold_backward.default(getitem_522, detach_261, 0);  getitem_522 = detach_261 = None
        native_batch_norm_backward_43 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_40, convolution_9, primals_29, getitem_50, getitem_51, getitem_48, getitem_49, True, 1e-05, [True, True, True]);  threshold_backward_40 = convolution_9 = primals_29 = getitem_50 = getitem_51 = getitem_48 = getitem_49 = None
        getitem_525 = native_batch_norm_backward_43[0]
        getitem_526 = native_batch_norm_backward_43[1]
        getitem_527 = native_batch_norm_backward_43[2];  native_batch_norm_backward_43 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(getitem_525, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_525 = primals_28 = None
        getitem_528 = convolution_backward_43[0]
        getitem_529 = convolution_backward_43[1];  convolution_backward_43 = None
        _to_copy_44 = torch.ops.aten._to_copy.default(getitem_529, dtype = torch.float32);  getitem_529 = None
        detach_264 = torch.ops.aten.detach.default(relu_7);  relu_7 = None
        detach_265 = torch.ops.aten.detach.default(detach_264);  detach_264 = None
        threshold_backward_41 = torch.ops.aten.threshold_backward.default(getitem_528, detach_265, 0);  getitem_528 = detach_265 = None
        native_batch_norm_backward_44 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_41, convolution_8, primals_26, getitem_45, getitem_46, getitem_43, getitem_44, True, 1e-05, [True, True, True]);  threshold_backward_41 = convolution_8 = primals_26 = getitem_45 = getitem_46 = getitem_43 = getitem_44 = None
        getitem_531 = native_batch_norm_backward_44[0]
        getitem_532 = native_batch_norm_backward_44[1]
        getitem_533 = native_batch_norm_backward_44[2];  native_batch_norm_backward_44 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(getitem_531, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_531 = primals_25 = None
        getitem_534 = convolution_backward_44[0]
        getitem_535 = convolution_backward_44[1];  convolution_backward_44 = None
        _to_copy_45 = torch.ops.aten._to_copy.default(getitem_535, dtype = torch.float32);  getitem_535 = None
        add_82 = torch.ops.aten.add.Tensor(threshold_backward_39, getitem_534);  threshold_backward_39 = getitem_534 = None
        detach_268 = torch.ops.aten.detach.default(relu_6);  relu_6 = None
        detach_269 = torch.ops.aten.detach.default(detach_268);  detach_268 = None
        threshold_backward_42 = torch.ops.aten.threshold_backward.default(add_82, detach_269, 0);  add_82 = detach_269 = None
        native_batch_norm_backward_45 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_42, convolution_7, primals_23, getitem_40, getitem_41, getitem_38, getitem_39, True, 1e-05, [True, True, True]);  convolution_7 = primals_23 = getitem_40 = getitem_41 = getitem_38 = getitem_39 = None
        getitem_537 = native_batch_norm_backward_45[0]
        getitem_538 = native_batch_norm_backward_45[1]
        getitem_539 = native_batch_norm_backward_45[2];  native_batch_norm_backward_45 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(getitem_537, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_537 = primals_22 = None
        getitem_540 = convolution_backward_45[0]
        getitem_541 = convolution_backward_45[1];  convolution_backward_45 = None
        _to_copy_46 = torch.ops.aten._to_copy.default(getitem_541, dtype = torch.float32);  getitem_541 = None
        detach_272 = torch.ops.aten.detach.default(relu_5);  relu_5 = None
        detach_273 = torch.ops.aten.detach.default(detach_272);  detach_272 = None
        threshold_backward_43 = torch.ops.aten.threshold_backward.default(getitem_540, detach_273, 0);  getitem_540 = detach_273 = None
        native_batch_norm_backward_46 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_43, convolution_6, primals_20, getitem_35, getitem_36, getitem_33, getitem_34, True, 1e-05, [True, True, True]);  threshold_backward_43 = convolution_6 = primals_20 = getitem_35 = getitem_36 = getitem_33 = getitem_34 = None
        getitem_543 = native_batch_norm_backward_46[0]
        getitem_544 = native_batch_norm_backward_46[1]
        getitem_545 = native_batch_norm_backward_46[2];  native_batch_norm_backward_46 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(getitem_543, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_543 = primals_19 = None
        getitem_546 = convolution_backward_46[0]
        getitem_547 = convolution_backward_46[1];  convolution_backward_46 = None
        _to_copy_47 = torch.ops.aten._to_copy.default(getitem_547, dtype = torch.float32);  getitem_547 = None
        detach_276 = torch.ops.aten.detach.default(relu_4);  relu_4 = None
        detach_277 = torch.ops.aten.detach.default(detach_276);  detach_276 = None
        threshold_backward_44 = torch.ops.aten.threshold_backward.default(getitem_546, detach_277, 0);  getitem_546 = detach_277 = None
        native_batch_norm_backward_47 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_44, convolution_5, primals_17, getitem_30, getitem_31, getitem_28, getitem_29, True, 1e-05, [True, True, True]);  threshold_backward_44 = convolution_5 = primals_17 = getitem_30 = getitem_31 = getitem_28 = getitem_29 = None
        getitem_549 = native_batch_norm_backward_47[0]
        getitem_550 = native_batch_norm_backward_47[1]
        getitem_551 = native_batch_norm_backward_47[2];  native_batch_norm_backward_47 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(getitem_549, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_549 = primals_16 = None
        getitem_552 = convolution_backward_47[0]
        getitem_553 = convolution_backward_47[1];  convolution_backward_47 = None
        _to_copy_48 = torch.ops.aten._to_copy.default(getitem_553, dtype = torch.float32);  getitem_553 = None
        add_83 = torch.ops.aten.add.Tensor(threshold_backward_42, getitem_552);  threshold_backward_42 = getitem_552 = None
        detach_280 = torch.ops.aten.detach.default(relu_3);  relu_3 = None
        detach_281 = torch.ops.aten.detach.default(detach_280);  detach_280 = None
        threshold_backward_45 = torch.ops.aten.threshold_backward.default(add_83, detach_281, 0);  add_83 = detach_281 = None
        native_batch_norm_backward_48 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_45, convolution_4, primals_14, getitem_25, getitem_26, getitem_23, getitem_24, True, 1e-05, [True, True, True]);  convolution_4 = primals_14 = getitem_25 = getitem_26 = getitem_23 = getitem_24 = None
        getitem_555 = native_batch_norm_backward_48[0]
        getitem_556 = native_batch_norm_backward_48[1]
        getitem_557 = native_batch_norm_backward_48[2];  native_batch_norm_backward_48 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(getitem_555, getitem_5, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_555 = primals_13 = None
        getitem_558 = convolution_backward_48[0]
        getitem_559 = convolution_backward_48[1];  convolution_backward_48 = None
        _to_copy_49 = torch.ops.aten._to_copy.default(getitem_559, dtype = torch.float32);  getitem_559 = None
        native_batch_norm_backward_49 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_45, convolution_3, primals_11, getitem_20, getitem_21, getitem_18, getitem_19, True, 1e-05, [True, True, True]);  threshold_backward_45 = convolution_3 = primals_11 = getitem_20 = getitem_21 = getitem_18 = getitem_19 = None
        getitem_561 = native_batch_norm_backward_49[0]
        getitem_562 = native_batch_norm_backward_49[1]
        getitem_563 = native_batch_norm_backward_49[2];  native_batch_norm_backward_49 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(getitem_561, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_561 = primals_10 = None
        getitem_564 = convolution_backward_49[0]
        getitem_565 = convolution_backward_49[1];  convolution_backward_49 = None
        _to_copy_50 = torch.ops.aten._to_copy.default(getitem_565, dtype = torch.float32);  getitem_565 = None
        detach_284 = torch.ops.aten.detach.default(relu_2);  relu_2 = None
        detach_285 = torch.ops.aten.detach.default(detach_284);  detach_284 = None
        threshold_backward_46 = torch.ops.aten.threshold_backward.default(getitem_564, detach_285, 0);  getitem_564 = detach_285 = None
        native_batch_norm_backward_50 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_46, convolution_2, primals_8, getitem_15, getitem_16, getitem_13, getitem_14, True, 1e-05, [True, True, True]);  threshold_backward_46 = convolution_2 = primals_8 = getitem_15 = getitem_16 = getitem_13 = getitem_14 = None
        getitem_567 = native_batch_norm_backward_50[0]
        getitem_568 = native_batch_norm_backward_50[1]
        getitem_569 = native_batch_norm_backward_50[2];  native_batch_norm_backward_50 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(getitem_567, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_567 = primals_7 = None
        getitem_570 = convolution_backward_50[0]
        getitem_571 = convolution_backward_50[1];  convolution_backward_50 = None
        _to_copy_51 = torch.ops.aten._to_copy.default(getitem_571, dtype = torch.float32);  getitem_571 = None
        detach_288 = torch.ops.aten.detach.default(relu_1);  relu_1 = None
        detach_289 = torch.ops.aten.detach.default(detach_288);  detach_288 = None
        threshold_backward_47 = torch.ops.aten.threshold_backward.default(getitem_570, detach_289, 0);  getitem_570 = detach_289 = None
        native_batch_norm_backward_51 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_47, convolution_1, primals_5, getitem_10, getitem_11, getitem_8, getitem_9, True, 1e-05, [True, True, True]);  threshold_backward_47 = convolution_1 = primals_5 = getitem_10 = getitem_11 = getitem_8 = getitem_9 = None
        getitem_573 = native_batch_norm_backward_51[0]
        getitem_574 = native_batch_norm_backward_51[1]
        getitem_575 = native_batch_norm_backward_51[2];  native_batch_norm_backward_51 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(getitem_573, getitem_5, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_573 = getitem_5 = primals_4 = None
        getitem_576 = convolution_backward_51[0]
        getitem_577 = convolution_backward_51[1];  convolution_backward_51 = None
        _to_copy_52 = torch.ops.aten._to_copy.default(getitem_577, dtype = torch.float32);  getitem_577 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_558, getitem_576);  getitem_558 = getitem_576 = None
        max_pool2d_with_indices_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(add_84, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_6);  add_84 = getitem_6 = None
        detach_292 = torch.ops.aten.detach.default(relu);  relu = None
        detach_293 = torch.ops.aten.detach.default(detach_292);  detach_292 = None
        threshold_backward_48 = torch.ops.aten.threshold_backward.default(max_pool2d_with_indices_backward, detach_293, 0);  max_pool2d_with_indices_backward = detach_293 = None
        native_batch_norm_backward_52 = torch.ops.aten.native_batch_norm_backward.default(threshold_backward_48, convolution, primals_2, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  threshold_backward_48 = convolution = primals_2 = getitem_3 = getitem_4 = getitem_1 = getitem_2 = None
        getitem_579 = native_batch_norm_backward_52[0]
        getitem_580 = native_batch_norm_backward_52[1]
        getitem_581 = native_batch_norm_backward_52[2];  native_batch_norm_backward_52 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(getitem_579, primals_321, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  getitem_579 = primals_321 = primals_1 = None
        getitem_583 = convolution_backward_52[1];  convolution_backward_52 = None
        _to_copy_53 = torch.ops.aten._to_copy.default(getitem_583, dtype = torch.float32);  getitem_583 = None
        return [_to_copy_53, getitem_580, getitem_581, _to_copy_52, getitem_574, getitem_575, _to_copy_51, getitem_568, getitem_569, _to_copy_50, getitem_562, getitem_563, _to_copy_49, getitem_556, getitem_557, _to_copy_48, getitem_550, getitem_551, _to_copy_47, getitem_544, getitem_545, _to_copy_46, getitem_538, getitem_539, _to_copy_45, getitem_532, getitem_533, _to_copy_44, getitem_526, getitem_527, _to_copy_43, getitem_520, getitem_521, _to_copy_42, getitem_514, getitem_515, _to_copy_41, getitem_508, getitem_509, _to_copy_40, getitem_502, getitem_503, _to_copy_39, getitem_496, getitem_497, _to_copy_38, getitem_490, getitem_491, _to_copy_37, getitem_484, getitem_485, _to_copy_36, getitem_478, getitem_479, _to_copy_35, getitem_472, getitem_473, _to_copy_34, getitem_466, getitem_467, _to_copy_33, getitem_460, getitem_461, _to_copy_32, getitem_454, getitem_455, _to_copy_31, getitem_448, getitem_449, _to_copy_30, getitem_442, getitem_443, _to_copy_29, getitem_436, getitem_437, _to_copy_28, getitem_430, getitem_431, _to_copy_27, getitem_424, getitem_425, _to_copy_26, getitem_418, getitem_419, _to_copy_25, getitem_412, getitem_413, _to_copy_24, getitem_406, getitem_407, _to_copy_23, getitem_400, getitem_401, _to_copy_22, getitem_394, getitem_395, _to_copy_21, getitem_388, getitem_389, _to_copy_20, getitem_382, getitem_383, _to_copy_19, getitem_376, getitem_377, _to_copy_18, getitem_370, getitem_371, _to_copy_17, getitem_364, getitem_365, _to_copy_16, getitem_358, getitem_359, _to_copy_15, getitem_352, getitem_353, _to_copy_14, getitem_346, getitem_347, _to_copy_13, getitem_340, getitem_341, _to_copy_12, getitem_334, getitem_335, _to_copy_11, getitem_328, getitem_329, _to_copy_10, getitem_322, getitem_323, _to_copy_9, getitem_316, getitem_317, _to_copy_8, getitem_310, getitem_311, _to_copy_7, getitem_304, getitem_305, _to_copy_6, getitem_298, getitem_299, _to_copy_5, getitem_292, getitem_293, _to_copy_4, getitem_286, getitem_287, _to_copy_3, getitem_280, getitem_281, _to_copy_2, getitem_274, getitem_275, _to_copy_1, getitem_268, getitem_269, t_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        
