# https://blog.finxter.com/pytest-a-complete-overview/?tl_inbound=1&tl_target_all=1&tl_form_type=1&tl_period_type=3
from src.model.Route import Route
from src.model.Request import Request
from src.model.Vehicle import Vehicle
import pytest

dist_matrix = {
    "O": {"A": 150, "B": 100},
    "A": {"A'": 300, "B": 25},
    "B": {"A'": 400, "B'": 300},
    "A'": {"B": 300, "B'": 25},
}

"""
      route = [1, 2 , 1', 2']
route ids   = [1, 2 , 3 , 4 ]

    arr.:              150         175          475          500
   route: [O]---150--->[A]---25--->[B]---400--->[A']---25--->[B']
      tw:           [0  ,180)   [20 ,200)    [300,600)    [320,620)
 e. arr.:              150         100          450          400
  
"""


@pytest.fixture
def v1():
    return Vehicle("O", 1, alias="V1")


@pytest.fixture
def v2():
    return Vehicle("O", 2, alias="V2")


@pytest.fixture
def r1():
    return Request("A", "A'", 0, 180, 300, 600, alias="A")


@pytest.fixture
def r2():
    return Request("B", "B'", 20, 200, 320, 620, alias="B")


@pytest.fixture
def valid_node_sequence(r1, r2):
    """
       arr.:              150         175          575          600
      route: [O]---150--->[A]---25--->[B]---400--->[A']---25--->[B']
         tw:           [0  ,180)   [20 ,200)    [300,600)    [320,620)
    e. arr.:              150         100          450          400
    """

    return [r1.pickup_node, r2.pickup_node, r1.dropoff_node, r2.dropoff_node]


@pytest.fixture
def invalid_node_sequence(r1, r2):
    """
       arr.:              150         175          350         375
      route: [O]---150--->[A]--300--->[A']---50--->[B]---25--->[B']
         tw:           [0  ,180)   [300,600)    [20 ,200)   [320,620)
    e. arr.:              150         450          100         400
    """

    return [r1.pickup_node, r1.dropoff_node, r2.pickup_node, r2.dropoff_node]


def test_feasible_route(valid_node_sequence, v2):
    v2.visit_nodes(*valid_node_sequence)
    assert v2.route.is_feasible(distance_matrix=dist_matrix) == True


def test_infeasible_route(invalid_node_sequence, v2):
    v2.visit_nodes(*invalid_node_sequence)
    assert v2.route.is_feasible(distance_matrix=dist_matrix) == False


def test_visit_nodes(valid_node_sequence, v2):
    v2.visit_nodes(*valid_node_sequence)
    assert all(n in v2.route.nodes for n in valid_node_sequence)


def test_assign(r1, v2):
    v2.assign(r1)
    assert r1 in v2.requests
