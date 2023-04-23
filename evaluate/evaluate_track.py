import pandas as pd
import motmetrics as mm

# 读入 predict track 和 ground truth csv 文件
# predict_file = r"G:\20x_dataset\evaluate_data\split-copy19\group0\track-GT.csv"
# ground_truth_file = r"G:\20x_dataset\evaluate_data\split-copy19\group0\tracking_output\track.csv"

# predict_file = r"G:\20x_dataset\evaluate_data\src06\trackmeta.csv"
# predict_file = r"G:\20x_dataset\evaluate_data\src06\track\track.csv"
predict_file = r"E:\paper\evaluate_data\src06\tracking_output\track.csv"
ground_truth_file = r"E:\paper\evaluate_data\src06\track-GT.csv"

# 从CSV文件中读取跟踪和真实轨迹
predict_df = pd.read_csv(predict_file)
truth_df = pd.read_csv(ground_truth_file)

predict_df = predict_df.sort_values(by=['track_id', 'cell_id', 'frame_index'])

truth_df = truth_df.sort_values(by=['track_id', 'cell_id', 'frame_index'])


def evaluate(truth_df, predict_df):
    # Create an accumulator that will be used to accumulate errors.
    # It accepts two arguments: the list of metric names to compute, and whether the metrics are "single object" metrics
    # (meaning they're computed on a per-object basis, like tracking accuracy or recall), or "tracking metrics" (which consider
    # the tracking as a whole and measure things like identity switches or fragmentation)
    # metric_names = ['recall', 'precision', 'num_false_positives', 'num_misses', 'mota', 'motp', 'idf1']
    metric_names = ['num_frames', 'idf1', 'idp', 'idr',
                    'recall', 'precision', 'num_objects',
                    'mostly_tracked', 'partially_tracked',
                    'mostly_lost', 'num_false_positives',
                    'num_misses', 'num_switches',
                    'num_fragmentations', 'mota',
                    ]

    acc = mm.MOTAccumulator()

    # Update the accumulator for each frame in the sequence. In this example we assume the two dataframes (gt and dt)
    # have the same number of rows and are in the same order. If this is not the case you will need to perform some
    # sort of alignment of the dataframes (for example, sorting them by frame index and track ID) before calling update.
    for i, (frame_gt, frame_dt) in enumerate(zip(truth_df.groupby('frame_index'), predict_df.groupby('frame_index'))):
        _, gt_group = frame_gt
        _, dt_group = frame_dt

        # The update() function takes four arrays:
        # - oids (the list of IDs of ground truth objects present in the frame)
        # - hids (the list of IDs of detected objects present in the frame)
        # - dists (a 2D array with shape [len(oids), len(hids)] containing the pairwise distances between the ground truth and
        #         detected objects)
        # - frameid (an optional frame ID that can be used to identify the frame in case the dataframes aren't sorted by frame)
        # In this example we assume that the track IDs in the ground truth and detection dataframes are the same.
        # If the track IDs don't match, you will need to perform some kind of matching or linking step before calling update().
        oids = gt_group['cell_id'].values
        hids = dt_group['cell_id'].values
        dists = mm.distances.norm2squared_matrix(gt_group[['center_x', 'center_y']].values,
                                                 dt_group[['center_x', 'center_y']].values)

        acc.update(oids, hids, dists, frameid=i)

    # Compute metrics on the accumulated errors.
    # The compute_metrics() function returns a dictionary with the computed metrics.
    # The first argument is the accumulator to compute the metrics on. The second argument is the metric to compute. It
    # can be a string (e.g. "mota") or a list of strings (e.g. ["mota", "num_false_positives"]).
    metrics = mm.metrics.create()
    summary = metrics.compute(acc, metrics=metric_names)
    # summary = metrics.compute(acc, metrics=list(mm.metrics.motchallenge_metrics))

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll',
                 'precision': 'Prcn', 'num_objects': 'GT',
                 'mostly_tracked': 'MT', 'partially_tracked': 'PT',
                 'mostly_lost': 'ML', 'num_false_positives': 'FP',
                 'num_misses': 'FN', 'num_switches': 'IDsw',
                 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP',
                 }
    )
    print(strsummary)

    # Print the summary of metrics
    print(summary)
    print(type(summary))
    print(mm.io.render_summary(summary, formatters=metrics.formatters, namemap=mm.io.motchallenge_metric_names))
    # summary.to_csv(r'G:\20x_dataset\evaluate_data\split-copy19\group0\track_evaluate.csv')


evaluate(truth_df, predict_df)
