"""Run manually, e.g.:
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 \
           --headless -u 20 -r 5 -t 60s --csv results/locust
"""
import random

from locust import HttpUser, between, task


class SensorUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def predict(self):
        payload = {k: random.gauss(0, 1) for k in
                   ("temperature", "vibration", "pressure", "humidity", "current", "rpm")}
        self.client.post("/predict", json=payload)
