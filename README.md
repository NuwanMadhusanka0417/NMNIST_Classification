## Branch explanation

This graph generate graph using SOTA method for generation. This specific generation function is located in /src/events_to_graph_converter.py 
Paper - Neuromorphic Imaging and Classification with Graph Learning (http://arxiv.org/abs/2309.15627)
_________________________________________________________________
## Event Data reading

To read event data, tonic library is used. That library can download data if data is not available 

```python
train_ds = tonic.datasets.NMNIST(
    save_to=data_path,        # or just data_path as positional
    train=True,
    transform=voxelise,
    download=True            # recognised in ≥ v1.5-dev
)
```

When we load data, wwe can choose the data reoresentation events. for example,

1. ToAveragedTimesurface
2. ToFrame
3. ToSparseTensor
4. ToImage
5. ToTimesurface
6. ToVoxelGrid
7. ToBinaRep

We can choose any type according to our project approach and modify following line according to this,
 
```python
voxelise = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=6) 
```

I we don't give any transform, functin will return raw data (Structured array).

### Voxlization with function

This function provide voxelization method. But it divide only time frame into voxels, and normalize time considering whole time frame.
That method is not compatible;e with our approach.
________________________________________________________________________

## Code explanation

This code contain two parts
1. graph feneration
2. graph - HV convertion and classification

### Graph generation.

To graph generation 

```python
python3 graph_generation.py
```

In this code, there are two parametrerss.

```python
normalized_feat = False
num_of_graph_events = 50 
```

normalized_feat parameter should be tru if we use generated graoh in GNN. othervise keep it False.  num_of_graph_events describe, how many events used for graph generation. Because, each sample dta have aroun 5000 events. So processing all events are costly. Therefore, program limit number of events used in graph.


### graph - HV convertion and classification

For HV convertion and classification, you can use __main.py__ or __main.ipynb__ file. For HV convertion, this program uze GVFA.

