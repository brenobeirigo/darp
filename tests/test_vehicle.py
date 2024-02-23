from src.model.Vehicle import Vehicle
from shapely.geometry import Point
import math


def test_vehicle_creation_closed_trip_to_origin():
    Vehicle.cleanup()
    v = Vehicle(1, 4, origin_earliest_time=4, alias="CAR")
    assert v.alias == "CAR"
    assert v.capacity == 4
    assert v.origin_node.point == None
    assert v.origin_node.pos == 1
    assert v.origin_node.alias == "O(CAR)"
    assert v.origin_node.arrival == 4
    assert v.origin_node.departure == None
    assert v.open_trip == False
    assert v.destination_node.pos == 1
    assert v.destination_node.point == None
    assert v.destination_node.alias == "D(CAR)"
    assert v.destination_node.arrival == None
    assert v.destination_node.departure == None


def test_vehicle_creation_closed_trip_to_other_destination():
    Vehicle.cleanup()
    v = Vehicle(1, 4, destination_id=2, origin_earliest_time=4, alias="CAR")
    assert v.alias == "CAR"
    assert v.capacity == 4
    assert v.origin_node.point == None
    assert v.origin_node.alias == "O(CAR)"
    assert v.origin_node.pos == 1
    assert v.origin_node.arrival == 4
    assert v.origin_node.departure == None
    assert v.open_trip == False
    assert v.destination_node.pos == 2
    assert v.destination_node.point == None
    assert v.destination_node.alias == "D(CAR)"
    assert v.destination_node.arrival == None
    assert v.destination_node.departure == None


def test_vehicle_creation_open_trip():
    Vehicle.cleanup()
    v = Vehicle(1, 4, open_trip=True)
    assert v.alias == "V1"
    assert v.capacity == 4
    assert v.origin_node.point == None
    assert v.origin_node.alias == "O(1)"
    assert v.origin_node.arrival == 0
    assert v.origin_node.departure == None
    assert v.open_trip == True
    assert v.destination_node == None
