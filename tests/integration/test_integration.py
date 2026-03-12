import pytest
import os
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from pathlib import Path 
from escoplus_skills_extender import app

load_dotenv()

client = TestClient(app)

CREDENTIALS_PRESENT = os.getenv("TRACKER_USERNAME") and os.getenv("TRACKER_PASSWORD")

@pytest.fixture(autouse=True)
def cleanup_completed_analyses():
    """Delete any cache files created under Completed_Analyses/ during each test."""
    folder = Path("Completed_Analyses")
    files_before = set(folder.glob("*.json")) if folder.exists() else set()
 
    yield
 
    if folder.exists():
        files_after = set(folder.glob("*.json"))
        for new_file in files_after - files_before:
            new_file.unlink()

@pytest.fixture(scope="class", autouse=True)
def cleanup_generated_csvs():
    """Runs once after all tests in the class have finished."""
    yield  # This is where the tests happen
    
    # After tests are done, find and remove the generated files
    files_to_remove = [
        "ESCOplus_Extended_from_Policies.csv",
        "ESCOplus_Extended_from_Profiles.csv",
        "ESCOplus_Extended_from_Jobs.csv",
        "ESCOplus_Extended_from_Courses.csv"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"\n Cleaned up {file}")
            except Exception as e:
                print(f"\n Could not remove {file}: {e}")


@pytest.mark.skipif(not CREDENTIALS_PRESENT, reason="Tracker credentials missing in environment/env")
class TestESCOPlusIntegration:

    def test_law_policies_extension_connectivity(self):
        """
        Tests /api/analysis/law-policies_extend_esco.
        Verifies real API auth, document retrieval, and network generation.
        """
        response = client.get(
            "/api/analysis/law-policies_extend_esco",
            params={
                "keywords": "ai",
                "source": "eur_lex",
                "similarity_threshold": 0.8,  # Lowered for test to ensure matches
                "confidence_threshold": 0.6
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check for core keys in the response
        assert "summary" in data
        assert "network" in data
        assert "network_stats" in data
        assert data["summary"]["Policies processed"] >= 0


    def test_profiles_extension_connectivity(self):
        """
        Tests /api/analysis/profiles_extend_esco.
        Verifies profile fetching and similarity metrics.
        """
        response = client.get(
            "/api/analysis/profiles_extend_esco",
            params={
                "keywords": "software",
                "max_pages": 1,  # Keep it fast
                "similarity_threshold": 0.8
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "explainability_metrics" in data
        assert "network" in data
        assert "summary" in data


    def test_jobs_ultra_extension_connectivity(self):
        """
        Tests /api/analysis/jobs_ultra.
        Verifies job fetching and ESCO+ proposed extensions.
        """
        response = client.get(
            "/api/analysis/jobs_ultra",
            params={
                "occupation_ids": "http://data.europa.eu/esco/isco/C3133",
                "similarity_threshold": 0.8
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "proposed_extensions" in data
        assert "network" in data
        assert data["summary"]["Job postings processed"] >= 0


    def test_forecasting_link_prediction(self):
        """
        Tests /api/forecasting/profiles.
        Verifies the Link Prediction (Adamic-Adar/Jaccard) logic on the real graph.
        """
        response = client.get(
            "/api/forecasting/profiles",
            params={
                "keywords": "ai",
                "method": "adamic_adar",
                "top_k": 5,
                "similarity_threshold": 0.4
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "predicted_links" in data
        assert "summary" in data
        assert data["summary"]["Method used"] == "adamic_adar"


    def test_courses_extension_connectivity(self):
        """
        Tests /api/analysis/courses_ultra.
        Verifies course fetching and skill mapping.
        """
        response = client.get(
            "/api/analysis/courses_ultra",
            params={
                "keywords": "green",
                "similarity_threshold": 0.4
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "network" in data