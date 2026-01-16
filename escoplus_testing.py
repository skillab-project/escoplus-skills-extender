import unittest
from fastapi.testclient import TestClient

# 👇 IMPORT YOUR APP
# this must match the filename where the FastAPI app lives
from escoplus_skills_extender import app



class TestESCOPlusAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # --------------------------------------------------
    # BASIC APP CHECK
    # --------------------------------------------------
    def test_app_is_alive(self):
        response = self.client.get("/")
        # Root endpoint not defined → 404 is OK
        self.assertIn(response.status_code, [200, 404])

    # --------------------------------------------------
    # ANALYSIS ENDPOINTS (ESCO EXTENSION)
    # --------------------------------------------------
    def test_law_policies_extend_esco(self):
        response = self.client.get(
            "/api/analysis/law-policies_extend_esco",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    def test_profiles_extend_esco(self):
        response = self.client.get(
            "/api/analysis/profiles_extend_esco",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    def test_jobs_ultra(self):
        response = self.client.get(
            "/api/analysis/jobs_ultra",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    def test_courses_ultra(self):
        response = self.client.get(
            "/api/analysis/courses_ultra",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    # --------------------------------------------------
    # FORECASTING / LINK PREDICTION ENDPOINTS
    # --------------------------------------------------
    def test_profiles_link_prediction(self):
        response = self.client.get(
            "/api/forecasting/profiles",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    def test_ku_link_prediction(self):
        response = self.client.get(
            "/api/forecasting/ku-link-prediction"
        )
        self.assertIn(response.status_code, [200, 422])

    def test_jobs_link_prediction(self):
        response = self.client.get(
            "/api/forecasting/jobsd-forecast",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    def test_courses_link_prediction(self):
        response = self.client.get(
            "/api/forecasting/courses",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])

    def test_law_link_prediction(self):
        response = self.client.get(
            "/api/forecasting/law_predict",
            params={"keywords": "ai"}
        )
        self.assertIn(response.status_code, [200, 422])


if __name__ == "__main__":
    unittest.main()
