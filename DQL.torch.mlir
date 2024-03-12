module @dql_test {
  func.func @test_dynamicquantizelinear(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.aten.max %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[],f32>
    %1 = torch.aten.min %arg0 : !torch.vtensor<[6],f32> -> !torch.vtensor<[],f32>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %3 = torch.aten.full %2, %float0.000000e00, %int6, %none_0, %none_0, %none_0 : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.maximum %0, %3 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %5 = torch.aten.minimum %1, %3 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %6 = torch.aten.sub.Tensor %4, %5, %int1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
    %float2.550000e02 = torch.constant.float 2.550000e+02
    %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_1 = torch.constant.none
    %int6_2 = torch.constant.int 6
    %8 = torch.aten.full %7, %float2.550000e02, %int6_2, %none_1, %none_1, %none_1 : !torch.list<int>, !torch.float, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.div.Tensor %6, %8 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %10 = torch.aten.div.Tensor %1, %9 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %11 = torch.aten.sub.Tensor %3, %10, %int1 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[],f32>
    %12 = torch.aten.clamp %11, %float0.000000e00, %float2.550000e02 : !torch.vtensor<[],f32>, !torch.float, !torch.float -> !torch.vtensor<[],f32>
    %13 = torch.aten.round %12 : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %int13 = torch.constant.int 13
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %14 = torch.aten.to.dtype %13, %int13, %false, %false, %none_3 : !torch.vtensor<[],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[],ui8>
    %15 = torch.aten.item %14 : !torch.vtensor<[],ui8> -> !torch.int
    %16 = torch.aten.item %9 : !torch.vtensor<[],f32> -> !torch.float
    %17 = torch.aten.quantize_per_tensor %arg0, %16, %15, %int13 : !torch.vtensor<[6],f32>, !torch.float, !torch.int, !torch.int -> !torch.vtensor<[6],!torch.quint8>
    %18 = torch.aten.int_repr %17 : !torch.vtensor<[6],!torch.quint8> -> !torch.vtensor<[6],ui8>
    return %18, %9, %14 : !torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>
  }
}

