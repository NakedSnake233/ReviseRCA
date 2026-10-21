"""
Locust load generator for TrainTicket.
Simulates realistic user behavior: search trains, book, view orders, cancel, rebook.
"""

import random
import json
from locust import HttpUser, task, between

# Station names for search
STATIONS = [
    "Shang Hai", "Bei Jing", "Nan Jing", "Su Zhou", "Tai Yuan",
    "Shi Jia Zhuang", "Wu Han", "Xi An", "Zheng Zhou", "Ji Nan",
    "Hang Zhou", "Guang Zhou", "Chang Sha", "Tian Jin", "Cheng Du",
    "Chong Qing", "Gui Yang", "Kun Ming", "Nan Ning", "Fu Zhou",
]

# Login credentials (default user)
USER_ID = "4d2a08c0-81a5-4c18-8698-f4a4d104c68a"
USER_NAME = "fdse_microservice"
PASSWORD = "111111"


class TrainTicketUser(HttpUser):
    """Simulates a single TrainTicket user."""

    wait_time = between(1, 3)
    host = "http://ts-ui-dashboard:8080"

    def on_start(self):
        """Login to get a session token."""
        self.token = ""
        self.login()

    def login(self):
        """Login and store the bearer token."""
        with self.client.post(
            "/api/v1/users/login",
            json={"userName": USER_NAME, "password": PASSWORD},
            name="/login",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    self.token = data.get("data", {}).get("token", "")
                except Exception:
                    pass

    def _headers(self):
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _random_station_pair(self):
        """Get two different random stations."""
        dep, arr = random.sample(STATIONS, 2)
        return dep, arr

    @task(10)
    def search_trains(self):
        """Search for trains between two stations (most common action)."""
        dep, arr = self._random_station_pair()
        date = "2026-07-01"
        with self.client.post(
            "/api/v1/travelservice/trips/left",
            json={
                "startingPlace": dep,
                "endPlace": arr,
                "departureTime": date,
            },
            headers=self._headers(),
            name="/trips/left [search]",
            catch_response=True,
        ) as resp:
            pass

    @task(4)
    def search_high_speed(self):
        """Search for high-speed trains specifically."""
        dep, arr = self._random_station_pair()
        date = "2026-07-01"
        with self.client.post(
            "/api/v1/travel2service/trips/left",
            json={
                "startingPlace": dep,
                "endPlace": arr,
                "departureTime": date,
            },
            headers=self._headers(),
            name="/trips/left [high-speed]",
            catch_response=True,
        ) as resp:
            pass

    @task(3)
    def view_orders(self):
        """View user's order list."""
        with self.client.get(
            f"/api/v1/orderservice/order/{USER_ID}",
            headers=self._headers(),
            name="/order [GET]",
            catch_response=True,
        ) as resp:
            pass

    @task(3)
    def view_other_orders(self):
        """View other-type orders."""
        with self.client.get(
            f"/api/v1/orderOtherService/orderOther/{USER_ID}",
            headers=self._headers(),
            name="/orderOther [GET]",
            catch_response=True,
        ) as resp:
            pass

    @task(2)
    def view_travel_route(self):
        """Search for travel routes."""
        dep, arr = self._random_station_pair()
        with self.client.post(
            "/api/v1/routeplanservice/routePlan/cheapestRoute",
            json={"fromStationName": dep, "toStationName": arr},
            headers=self._headers(),
            name="/routePlan [cheapest]",
            catch_response=True,
        ) as resp:
            pass

    @task(1)
    def get_station_names(self):
        """Query station name list (triggers ts-station-service)."""
        with self.client.get(
            "/api/v1/stationservice/stations",
            headers=self._headers(),
            name="/stations [GET]",
            catch_response=True,
        ) as resp:
            pass

    @task(1)
    def get_contacts(self):
        """View contacts list (triggers ts-contacts-service)."""
        with self.client.get(
            f"/api/v1/contactservice/contacts/account/{USER_ID}",
            headers=self._headers(),
            name="/contacts [GET]",
            catch_response=True,
        ) as resp:
            pass
