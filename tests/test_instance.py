from src.data import parser
from testfixtures import TempDirectory
from src.model.Vehicle import Vehicle
from src.model.Request import Request
from src.instance.Instance import InstanceConfig
import os


def test_cordeau_parse():
    node_line = "1  -1.198  -5.164   3   1    0 1440"
    (
        id,
        x,
        y,
        service_duration,
        load,
        earliest,
        latest,
    ) = parser.parse_node_line(node_line)
    assert (
        id == 1
        and x == -1.198
        and y == -5.164
        and service_duration == 3
        and load == 1
        and earliest == 0
        and latest == 1440
    )


def test_cordeau_parse_request():
    Request.cleanup()
    pickup_node_line = "1  -1.198  -5.164   3   1    0 1440"
    dropoff_node_line = "17   6.687   6.731   4  -1  402  417"
    max_ride_time = 30

    r = parser.parse_request(
        pickup_node_line, dropoff_node_line, max_ride_time
    )

    assert (
        r.alias == "1"
        and r.load == 1
        and r.id == 1
        and r.pickup_tw.earliest == 0
        and r.pickup_tw.latest == 1440
        and r.dropoff_tw.earliest == 402
        and r.dropoff_tw.latest == 417
        and r.pickup_node.point.x == -1.198
        and r.pickup_node.point.y == -5.164
        and r.dropoff_node.point.x == 6.687
        and r.dropoff_node.point.y == 6.731
        and r.pickup_delay == 3
        and r.dropoff_delay == 4
        and r.max_ride_time == max_ride_time
    )


def test_cordeau_parse_vehicle():
    Vehicle.cleanup()
    origin_line = "0   0.000   0.000   0   0    0  480"
    destination_line = "33   0.000   0.000   0   0    0  480"
    vehicle_capacity = 3
    v = parser.parse_vehicle(origin_line, destination_line, vehicle_capacity)
    assert (
        v.origin_node.pos == 0
        and v.destination_node.pos == 33
        and v.origin_tw.earliest == 0
        and v.destination_tw.latest == 480
        and v.destination_tw.earliest == 0
        and v.origin_tw.latest == 480
        and v.capacity == 3
        and v.origin_node.point.x == 0.0
        and v.destination_node.point.y == 0.0
        and v.destination_node.point.x == 0.0
        and v.destination_node.point.y == 0.0
        and v.alias == "V0"
        and len(v.passengers) == 0
        and len(v.requests) == 0
        and v.origin_node.point.distance(v.destination_node.point) == 0
    )


def test_cordeau_parser():
    d = TempDirectory()

    content_instance = b"2 16 480 3 30\n  0   0.000   0.000   0   0    0  480\n  1  -1.198  -5.164   3   1    0 1440\n  2   5.573   7.114   3   1    0 1440\n  3  -6.614   0.072   3   1    0 1440\n  4  -7.374  -1.107   3   1    0 1440\n  5  -9.251   8.321   3   1    0 1440\n  6   6.498  -6.036   3   1    0 1440\n  7   0.861   6.903   3   1    0 1440\n  8   3.904  -5.261   3   1    0 1440\n  9   7.976  -9.000   3   1  276  291\n 10  -2.610   0.039   3   1   32   47\n 11   4.487   7.142   3   1  115  130\n 12   8.938  -4.388   3   1   14   29\n 13  -4.172  -9.096   3   1  198  213\n 14   7.835  -9.269   3   1  160  175\n 15   2.792  -7.944   3   1  180  195\n 16   5.212   9.271   3   1  366  381\n 17   6.687   6.731   3  -1  402  417\n 18  -2.192  -9.210   3  -1  322  337\n 19  -1.061   8.752   3  -1  179  194\n 20   6.883   0.882   3  -1  138  153\n 21   5.586  -1.554   3  -1   82   97\n 22  -9.865   1.398   3  -1   49   64\n 23  -9.800   5.697   3  -1  400  415\n 24   1.271   1.018   3  -1  298  313\n 25   4.404  -1.952   3  -1    0 1440\n 26   0.673   6.283   3  -1    0 1440\n 27   7.032   2.808   3  -1    0 1440\n 28  -0.694  -7.098   3  -1    0 1440\n 29   3.763  -7.269   3  -1    0 1440\n 30   6.634  -7.426   3  -1    0 1440\n 31  -9.450   3.792   3  -1    0 1440\n 32  -8.819  -4.749   3  -1    0 1440\n 33   0.000   0.000   0   0    0  480\n"
    filename = "a2-16"
    d.write(filename, content_instance)

    filepath = os.path.join(d.path, filename)
    instance = parser.parse_instance_from_filepath(
        filepath, instance_parser=parser.PARSER_TYPE_CORDEAU
    )
    print("InstanceX", instance)

    assert (
        instance.config_dict
        == InstanceConfig(
            n_vehicles=2,
            n_customers=16,
            time_horizon_min=480,
            vehicle_capacity=3,
            maximum_ride_time_min=30,
            # type=parser.PARSER_TYPE_CORDEAU,
            # path=filepath
        )
        and len(instance.requests) == 16
        and len(instance.vehicles) == 2
        and len(instance.nodes) == 36
        and instance.dist_matrix
        == {
            o.pos: {d.pos: o.point.distance(d.point) for d in instance.nodes}
            for o in instance.nodes
        }
    )

    assert len(instance.vehicles) > 0
