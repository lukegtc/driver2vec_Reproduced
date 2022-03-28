NUM_DRIVERS = 5


CHANNELS=["ACCELERATION", "ACCELERATION_PEDAL", "ACCELERATION_Y",
              "ACCELERATION_Z", "BRAKE_PEDAL", "CLUTCH_PEDAL", "CURVE_RADIUS",
              "DISTANCE", "DISTANCE_TO_NEXT_INTERSECTION",
              "DISTANCE_TO_NEXT_STOP_SIGNAL",
              "DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL",
              "DISTANCE_TO_NEXT_VEHICLE",
              "DISTANCE_TO_NEXT_YIELD_SIGNAL",
              "FAST_LANE", "GEARBOX",
              "HEADING", "HORN", "INDICATORS",
              "INDICATORS_ON_INTERSECTION",
              "LANE", "LANE_LATERAL_SHIFT_CENTER", "LANE_LATERAL_SHIFT_LEFT",
              "LANE_LATERAL_SHIFT_RIGHT", "LANE_WIDTH",
              "ROAD_ANGLE", "SPEED", 
              "SPEED_LIMIT", "SPEED_NEXT_VEHICLE", "SPEED_Y",
              "SPEED_Z", "STEERING_WHEEL"]

# Map group to list of columns
CHANNEL_GROUPS = {
                "acceleration":["ACCELERATION",
                                "ACCELERATION_PEDAL",
                                "ACCELERATION_Y",
                                "ACCELERATION_Z"],
                "speed":["SPEED",
                        "SPEED_NEXT_VEHICLE",
                        "SPEED_Y",
                        "SPEED_Z"],
                "distance":["DISTANCE",
                            "DISTANCE_TO_NEXT_INTERSECTION",
                            "DISTANCE_TO_NEXT_STOP_SIGNAL",
                            "DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL",
                            "DISTANCE_TO_NEXT_VEHICLE",
                            "DISTANCE_TO_NEXT_YIELD_SIGNAL"],
                "pedal":["BRAKE_PEDAL","CLUTCH_PEDAL"],
                "lane":["LANE",
                        "LANE_LATERAL_SHIFT_CENTER",
                        "LANE_LATERAL_SHIFT_LEFT",
                        "LANE_LATERAL_SHIFT_RIGHT",
                        "LANE_WIDTH",
                        "FAST_LANE"],

                "angle":["CURVE_RADIUS","ROAD_ANGLE","STEERING_WHEEL"],
                "indicator":["INDICATORS","INDICATORS_ON_INTERSECTION"],

 
                "gearbox":["GEARBOX"]
                        }


MAX_P_WAYS = 5

triplet_train_metrics = [('one_hot_accuracy', ()),
                         ('confusion_matrix', ()),
                         ('triplet_accuracy', ()),
                         ('triplet_ratio', ()),
                         ('triplet_diff_weight_ratio', ()),
                         ('tsne', ()),
                         ('tsne_collisions', ())]

# triplet_simple_eval_metrics = [('one_hot_accuracy', ()),
#                                ('confusion_matrix', ()),
#                                ('triplet_accuracy', ()),
#                                ('triplet_ratio', ()),
#                                ('triplet_diff_weight_ratio', ()),
#                                ('tsne', ()),
#                                ('tsne_collisions', ())]
# triplet_simple_eval_metrics.extend([('p_way_accuracy', (p, ))
#                                     for p in range(2, MAX_P_WAYS + 1, 1)])
#triplet_simple_eval_metrics.extend([('area_accuracy', (p, ))
#                                    for p in [-1, 1, 8, 9]])

triplet_lgbm_eval_metrics = [('one_hot_accuracy', ()),
                             ('confusion_matrix', ()),
                             # Only need to do these once
                             # ('triplet_accuracy', ()),
                             # ('triplet_ratio', ()),
                             # ('triplet_diff_weight_ratio', ()),
                             # ('tsne', ()),
                             # ('tsne_collisions', ())
                             ]
#triplet_lgbm_eval_metrics.extend([('per_driver_f1', (i, ))
#                                  for i in range(NUM_DRIVERS)]) 
triplet_lgbm_eval_metrics.extend([('p_way_accuracy', (p, ))
                                  for p in range(2, MAX_P_WAYS + 1, 1)])
TRIPLET_EVAL_METRICS = {'train': {'train': triplet_train_metrics},
                        'eval': {'eval_lgbm': (triplet_lgbm_eval_metrics,'lgbm_predict')
                                 },
                        'test': {'test_lgbm': (triplet_lgbm_eval_metrics,'lgbm_predict')}
                        }

