import pytest
import pandas as pd
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from escoplus_skills_extender import app

client = TestClient(app)

# ==========================================
# 1. MOCK DATA HELPER
# ==========================================

def get_mock_df():
    """Returns a clean DataFrame without using read_csv, to avoid mock recursion."""
    return pd.DataFrame({
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00"],
        "Title": ["Chief Executives", "Chief Executives"],
        "Example": ["Python", "FastAPI"],  # The critical column
        "Commodity Code": ["43232202", "43232306"],
        "Commodity Title": ["Software", "Software"],
        "Hot Technology": ["Y", "N"],
        "In Demand": ["N", "N"]
    })

# ==========================================
# 2. TESTS
# ==========================================

@patch("pandas.read_csv")
@patch("requests.post")
def test_law_policies_extend_esco(mock_post, mock_csv):
    """Test the Law/Policy extension logic with full API mocking."""
    
    # Force the mock to return our clean DataFrame
    mock_csv.return_value = get_mock_df()

    # 1. Mock Login Response
    mock_login = MagicMock()
    mock_login.text = '"fake_token"'
    mock_login.status_code = 200
    
    # 2. Mock Law-Policies Response
    mock_docs = MagicMock()
    mock_docs.json.return_value = {
        "items": [
            {"skills": ["http://esco/s1", "http://esco/s2"]},
            {"skills": ["http://esco/s1"]}
        ]
    }
    mock_docs.status_code = 200

    # 3. Mock ESCO Skills Label Lookup
    mock_skills = MagicMock()
    mock_skills.json.return_value = {
        "items": [
            {"id": "http://esco/s1", "label": "Document Management"},
            {"id": "http://esco/s2", "label": "Workflow Software"}
        ]
    }
    mock_skills.status_code = 200

    # Set side effects for the sequence of POST calls in the endpoint
    mock_post.side_effect = [mock_login, mock_docs, mock_skills]

    # Execute request - setting low similarity to ensure matches in tiny mock data
    response = client.get("/api/analysis/law-policies_extend_esco?keywords=data&similarity_threshold=0.1")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "summary" in data
    assert data["summary"]["Policies processed"] == 2
    assert "network" in data
    assert "network_stats" in data

@patch("pandas.read_csv")
@patch("requests.post")
def test_profiles_link_prediction(mock_post, mock_csv):
    """Test Forecasting Link Prediction logic."""
    
    mock_csv.return_value = get_mock_df()

    # Mock sequence: Login -> Profiles -> ESCO Skills
    mock_login = MagicMock()
    mock_login.text = '"token"'
    mock_login.status_code = 200

    mock_profiles = MagicMock()
    mock_profiles.json.return_value = {
        "items": [{"skills": ["http://esco/s1"]}]
    }
    mock_profiles.status_code = 200

    mock_esco = MagicMock()
    mock_esco.json.return_value = {
        "items": [{"id": "http://esco/s1", "label": "python"}] # Match mock CSV 'Python'
    }
    mock_esco.status_code = 200

    mock_post.side_effect = [mock_login, mock_profiles, mock_esco]

    # Execute request
    response = client.get("/api/forecasting/profiles?keywords=ai&method=jaccard&similarity_threshold=0.1")
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_links" in data
    assert data["summary"]["Method used"] == "jaccard"

def test_jobs_ultra_error_handling():
    """Test how the API handles a failed tracker authentication."""
    # Clean up any stale lock files from previous runs
    cache_file = Path("Completed_Analyses/completed_analysis_jobs_ultra_esco_test_sim0.8_conf0.6.json")
    cache_file.unlink(missing_ok=True)

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_post.return_value = mock_res

        response = client.get("/api/analysis/jobs_ultra?keywords=test")
        assert response.status_code == 200
        assert "error" in response.json()