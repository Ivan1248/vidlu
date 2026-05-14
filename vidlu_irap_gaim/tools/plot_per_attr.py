
import re
import matplotlib.pyplot as plt
from collections import defaultdict

LOG = """
```

```
1 val: 14899MiB, 0.0/s, loss=.0305, amF1=.5180, amP=.5690, amR=.5115, acc=.9005
```
> [!NOTE]- Per-attribute
> ```
> Attribute                                                    A     mF1
> ------------------------------------------------------  ------  ------
> Area type                                               0.8855  0.8774
> Bicycle facility                                        0.9991  0.4998
> Bicycle observed flow                                   0.9917  0.3320
> Curvature                                               0.6978  0.5227
> Delineation                                             0.9713  0.9573
> Grade                                                   0.9316  0.4952
> Intersection channelisation                             0.9964  0.4991
> Intersection quality                                    0.9771  0.3410
> Intersection type                                       0.9769  0.1441
> Land use - driver-side                                  0.7869  0.5556
> Land use - passenger-side                               0.7769  0.5982
> Lane width                                              0.9408  0.6027
> Median Type                                             0.9292  0.3576
> Number of lanes                                         0.9550  0.4351
> Paved shoulder - driver-side                            0.9432  0.6169
> Paved shoulder - passenger-side                         0.9374  0.4623
> Pedestrian crossing - inspected road                    0.9942  0.3526
> Pedestrian crossing - side road                         0.9978  0.3330
> Pedestrian crossing quality                             0.9933  0.3502
> Pedestrian observed flow across the road                0.9964  0.3933
> Pedestrian observed flow along the road driver-side     0.9642  0.2206
> Pedestrian observed flow along the road passenger-side  0.9537  0.2535
> Property access points                                  0.7860  0.5407
> Quality of curve                                        0.7339  0.6159
> Road condition                                          0.9031  0.5671
> Roadside severity - driver-side distance                0.7107  0.5461
> Roadside severity - driver-side object                  0.5820  0.4338
> Roadside severity - passenger-side distance             0.7500  0.5517
> Roadside severity - passenger-side object               0.6077  0.4522
> Roadworks                                               0.9946  0.8201
> School zone crossing supervisor                         0.9924  0.5781
> School zone warning                                     0.9924  0.5781
> Service road                                            0.9902  0.4975
> Sidewalk - driver-side                                  0.9165  0.3727
> Sidewalk - passanger-side                               0.9223  0.3410
> Sight distance                                          0.9092  0.7011
> Skid resistance / grip                                  0.9889  0.4935
> Speed management / traffic calming                      1.0000  1.0000
> Street lighting                                         0.8900  0.8838
> Upgrade cost                                            0.7620  0.5820
> Vehicle parking                                         0.8907  0.4819
> ```

```
2 val: 15141MiB, 0.0/s, loss=.0297, amF1=.5285, amP=.5824, amR=.5204, acc=.9034
```
> [!NOTE]- Per-attribute
> ```
> Attribute                                                    A     mF1
> ------------------------------------------------------  ------  ------
> Area type                                               0.8924  0.8857
> Bicycle facility                                        0.9998  0.5000
> Bicycle observed flow                                   0.9918  0.3496
> Curvature                                               0.7099  0.5584
> Delineation                                             0.9777  0.9678
> Grade                                                   0.9544  0.4883
> Intersection channelisation                             0.9964  0.4991
> Intersection quality                                    0.9748  0.3861
> Intersection type                                       0.9746  0.2077
> Land use - driver-side                                  0.7981  0.5711
> Land use - passenger-side                               0.7800  0.6105
> Lane width                                              0.9385  0.6387
> Median Type                                             0.9350  0.3516
> Number of lanes                                         0.9779  0.6258
> Paved shoulder - driver-side                            0.9421  0.6132
> Paved shoulder - passenger-side                         0.9316  0.4563
> Pedestrian crossing - inspected road                    0.9942  0.3705
> Pedestrian crossing - side road                         0.9980  0.3330
> Pedestrian crossing quality                             0.9933  0.3664
> Pedestrian observed flow across the road                0.9964  0.3328
> Pedestrian observed flow along the road driver-side     0.9644  0.2307
> Pedestrian observed flow along the road passenger-side  0.9583  0.2650
> Property access points                                  0.7938  0.5304
> Quality of curve                                        0.7368  0.6287
> Road condition                                          0.8628  0.4883
> Roadside severity - driver-side distance                0.7306  0.5664
> Roadside severity - driver-side object                  0.5976  0.4409
> Roadside severity - passenger-side distance             0.7548  0.5690
> Roadside severity - passenger-side object               0.6266  0.5014
> Roadworks                                               0.9938  0.7748
> School zone crossing supervisor                         0.9938  0.6121
> School zone warning                                     0.9938  0.6121
> Service road                                            0.9904  0.4976
> Sidewalk - driver-side                                  0.9065  0.3722
> Sidewalk - passanger-side                               0.9201  0.3301
> Sight distance                                          0.9092  0.7058
> Skid resistance / grip                                  0.9918  0.4804
> Speed management / traffic calming                      1.0000  1.0000
> Street lighting                                         0.8927  0.8860
> Upgrade cost                                            0.7686  0.5552
> Vehicle parking                                         0.8956  0.5108
> ```
>

```
3 val: 15141MiB, 0.0/s, loss=.0300, amF1=.5367, amP=.5840, amR=.5270, acc=.9060
```
> [!NOTE]- Per-attribute
> ```
> Attribute                                                    A     mF1
> ------------------------------------------------------  ------  ------
> Area type                                               0.8929  0.8871
> Bicycle facility                                        0.9996  0.4999
> Bicycle observed flow                                   0.9915  0.3319
> Curvature                                               0.7223  0.5446
> Delineation                                             0.9691  0.9538
> Grade                                                   0.9533  0.4881
> Intersection channelisation                             0.9958  0.5730
> Intersection quality                                    0.9753  0.3942
> Intersection type                                       0.9740  0.1887
> Land use - driver-side                                  0.7876  0.5405
> Land use - passenger-side                               0.7718  0.5938
> Lane width                                              0.9446  0.6408
> Median Type                                             0.9325  0.3713
> Number of lanes                                         0.9695  0.4575
> Paved shoulder - driver-side                            0.9494  0.6220
> Paved shoulder - passenger-side                         0.9416  0.4645
> Pedestrian crossing - inspected road                    0.9942  0.5363
> Pedestrian crossing - side road                         0.9973  0.4163
> Pedestrian crossing quality                             0.9933  0.4827
> Pedestrian observed flow across the road                0.9964  0.2950
> Pedestrian observed flow along the road driver-side     0.9653  0.2584
> Pedestrian observed flow along the road passenger-side  0.9579  0.2749
> Property access points                                  0.7829  0.5225
> Quality of curve                                        0.7477  0.6173
> Road condition                                          0.8800  0.4955
> Roadside severity - driver-side distance                0.7294  0.5728
> Roadside severity - driver-side object                  0.6226  0.4716
> Roadside severity - passenger-side distance             0.7593  0.5832
> Roadside severity - passenger-side object               0.6469  0.5162
> Roadworks                                               0.9946  0.8013
> School zone crossing supervisor                         0.9938  0.6289
> School zone warning                                     0.9938  0.6289
> Service road                                            0.9891  0.4973
> Sidewalk - driver-side                                  0.9120  0.3546
> Sidewalk - passanger-side                               0.9216  0.3354
> Sight distance                                          0.9221  0.6972
> Skid resistance / grip                                  0.9902  0.4397
> Speed management / traffic calming                      1.0000  1.0000
> Street lighting                                         0.8913  0.8825
> Upgrade cost                                            0.7767  0.5588
> Vehicle parking                                         0.9149  0.5848
> ```
"""

def parse_blocks(text):
    # Split on lines like "1 val:", "2 val:", ...
    blocks = re.split(r"\n\s*(\d+)\s+val:.*?\n", text)
    # re.split gives: [pre, step1, block1, step2, block2, ...]
    steps = []
    for i in range(1, len(blocks), 2):
        step = int(blocks[i])
        block = blocks[i+1]
        steps.append((step, block))
    return steps


def parse_attributes(block):
    data = {}

    for line in block.splitlines():
        line = line.strip()

        # Skip obvious non-data lines
        if (
            not line
            or line.startswith("Attribute")
            or set(line) <= {"-", " "}
        ):
            continue

        # Extract last two floats ONLY (robust to trailing junk)
        matches = re.findall(r"([0-9]+\.[0-9]+)", line)

        if len(matches) >= 2:
            A = float(matches[-2])
            mF1 = float(matches[-1])

            # Attribute name = everything before those numbers
            # Split from the right
            parts = re.split(r"\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+", line, maxsplit=1)
            if parts:
                name = parts[0].strip()
                data[name] = {"A": A, "mF1": mF1}

    return data

def collect(text):
    steps = parse_blocks(text)

    attr_series = defaultdict(lambda: {"A": {}, "mF1": {}})

    for step, block in steps:
        attrs = parse_attributes(block)
        for name, vals in attrs.items():
            attr_series[name]["A"][step] = vals["A"]
            attr_series[name]["mF1"][step] = vals["mF1"]

    return attr_series


def plot_metric(attr_series, metric="mF1", top_k=None, sort_by_last=True):
    # Convert to plottable format
    series = []

    for attr, metrics in attr_series.items():
        steps = sorted(metrics[metric].keys())
        values = [metrics[metric][s] for s in steps]
        if len(values) > 0:
            series.append((attr, steps, values))

    # Sort by last value
    if sort_by_last:
        series.sort(key=lambda x: x[2][-1])

    if top_k:
        series = series[:top_k]

    plt.figure(figsize=(10, 6))

    for attr, steps, values in series:
        plt.plot(steps, values, marker='o', label=attr)

    plt.xlabel("Validation step")
    plt.ylabel(metric)
    plt.title(f"Per-attribute {metric}")
    plt.grid(True)

    # Too many labels → only show legend if small
    plt.legend(fontsize=5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    attr_series = collect(LOG)

    # Plot worst 10 attributes by mF1
    plot_metric(attr_series, metric="mF1", top_k=21)

    # Plot best 10 attributes
    plot_metric(attr_series, metric="mF1", top_k=21, sort_by_last=False)