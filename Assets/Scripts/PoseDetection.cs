using System.Collections.Generic;
using Unity.Barracuda;
// using Unity.Sentis;
using UnityEngine;

namespace MediaPipe.BlazePose {

public sealed partial class PoseDetector : System.IDisposable
{

    ResourceSet _resources;
    ComputeBuffer _post1Buffer;
    ComputeBuffer _post2Buffer;
    ComputeBuffer _countBuffer;
    IWorker _worker;
    private Tensor _tensor;
    ComputeTensorData _tensorData;
    
    const int MaxDetection = 64;
    const int DetectionImageSize = 224;
        
    Detection[] _post2ReadCache;
    int[] _countReadCache = new int[1];
    
    public IEnumerable<Detection> Detections
      => _post2ReadCache ?? UpdatePost2ReadCache();
    
    public PoseDetector(ResourceSet resources)
    {
        _resources = resources;
        
        _post1Buffer = new ComputeBuffer
            (MaxDetection, Detection.Size, ComputeBufferType.Append);

        _post2Buffer = new ComputeBuffer
            (MaxDetection, Detection.Size, ComputeBufferType.Append);

        _countBuffer = new ComputeBuffer
            (1, sizeof(uint), ComputeBufferType.Raw);

        _worker = WorkerFactory.CreateWorker(ModelLoader.Load(_resources.detectionModel));
        // _worker = WorkerFactory.CreateWorker(BackendType.GPUCompute,ModelLoader.Load(_resources.detectionModel));
        
        var shape = new TensorShape(1, DetectionImageSize, DetectionImageSize, 3);
        // var shape = new TensorShape(1, 3, DetectionImageSize, DetectionImageSize);
        
        _tensorData = new ComputeTensorData(shape, "name", 0, false);
        _tensor = new Tensor(shape);
        // _tensorData = new ComputeTensorData(shape, false);
        // _tensor = TensorFloat.Zeros(shape);
        
        _tensor.AttachToDevice(_tensorData);
    }
    
    public void Dispose()
    {
        _tensor?.Dispose();
        _tensor = null;
        _tensorData = null;

        _post1Buffer?.Dispose();
        _post1Buffer = null;

        _post2Buffer?.Dispose();
        _post2Buffer = null;

        _countBuffer?.Dispose();
        _countBuffer = null;

        _worker?.Dispose();
        _worker = null;
    }

    public void ProcessImage(Texture image)
    {
        _post1Buffer.SetCounterValue(0);
        _post2Buffer.SetCounterValue(0);

        var pre = _resources.preprocess;
        var post1 = _resources.postprocess1;
        var post2 = _resources.postprocess2;
        
        pre.SetInt("_Size", DetectionImageSize);
        pre.SetVector("_Range", new Vector2(-1, 1));
        pre.SetTexture(0, "_Texture", image);
        pre.SetBuffer(0, "_Tensor", _tensorData.buffer);
        pre.Dispatch(0, DetectionImageSize / 8, DetectionImageSize / 8, 1);

        _worker.Execute(_tensor);

        var scoresRT = _worker.CopyOutputToTempRT("Identity_1",  1, 2254);
        var  boxesRT = _worker.CopyOutputToTempRT("Identity"  , 12, 2254);
        
        post1.SetInt("_RowOffset", 2254 - 1568);
        post1.SetTexture(0, "_Scores", scoresRT);
        post1.SetTexture(0, "_Boxes", boxesRT);
        post1.SetBuffer(0, "_Output", _post1Buffer);
        post1.Dispatch(0, 1, 1, 1);

        post1.SetInt("_RowOffset", 2254 - 1960);
        post1.SetTexture(1, "_Scores", scoresRT);
        post1.SetTexture(1, "_Boxes", boxesRT);
        post1.SetBuffer(1, "_Output", _post1Buffer);
        post1.Dispatch(1, 1, 1, 1);

        RenderTexture.ReleaseTemporary(scoresRT);
        RenderTexture.ReleaseTemporary(boxesRT);
        ComputeBuffer.CopyCount(_post1Buffer, _countBuffer, 0);
        
        post2.SetBuffer(0, "_Input", _post1Buffer);
        post2.SetBuffer(0, "_Count", _countBuffer);
        post2.SetBuffer(0, "_Output", _post2Buffer);
        post2.Dispatch(0, 1, 1, 1);

        ComputeBuffer.CopyCount(_post2Buffer, _countBuffer, 0);
        _post2ReadCache = null;
    }
    
    Detection[] UpdatePost2ReadCache()
    {
        _countBuffer.GetData(_countReadCache, 0, 0, 1);
        var count = _countReadCache[0];

        _post2ReadCache = new Detection[count];
        _post2Buffer.GetData(_post2ReadCache, 0, 0, count);

        return _post2ReadCache;
    }
}
} 
