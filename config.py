
# CONFIG FILE
network_size = 320
num_classes = 80

# ANCHOR BOXES
feature_map_numbers = 4
anchor_ranges = range(3, 7)
anchor_strides = [8, 16, 32, 64]
# anchor_areas = [32., 64., 128., 256.]
anchor_areas = [24., 48., 96., 192.]

default = [2 ** x for x in [0., 1./2.]]
anchor_scales = [default for i in range(feature_map_numbers)]
anchor_aspect_ratios = [[0.5, 1., 2.], [1./3., 0.5, 1., 2., 3.], [1./3., 0.5, 1., 2., 3.], [1./3., 0.5, 1., 2., 3.]]
anchor_per_grid = [len(anchor_scales[i]) * len(anchor_aspect_ratios[i]) for i in range(feature_map_numbers)]

# DECODE/ASSIGN BOXES
match_iou = 0.5
ignore_iou = 0.45
box_variances = [0.1, 0.1, 0.2, 0.2]

# INFERENEC MODE
confidence_threshold = 0.02
nms_iou_threshold = 0.45
max_predictions = 300



