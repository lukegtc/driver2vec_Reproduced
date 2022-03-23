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