"""
Custom MLServer Handler for Sklearn Models with ColumnTransformer

This handler allows clients to send raw JSON data and internally converts it
to pandas DataFrames for sklearn models that use ColumnTransformers.
"""

import joblib
import pandas as pd
import numpy as np
from typing import List, Optional, Any, Dict
from mlserver import MLModel
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    RequestOutput,
    ResponseOutput,
    Parameters
)
from mlserver.codecs import decode_args


class CustomSklearnModel(MLModel):
    """
    Custom MLServer model handler for sklearn models that need pandas input.
    
    This handler:
    1. Receives JSON data in MLServer format
    2. Converts to pandas DataFrame 
    3. Feeds to sklearn model with ColumnTransformer
    4. Returns predictions in MLServer format
    """

    async def load(self) -> bool:
        """Load the sklearn model from the specified URI"""
        model_uri = self.settings.parameters.uri
        print(f"Loading model from: {model_uri}")
        
        try:
            self.model = joblib.load(model_uri)
            print(f"Model loaded successfully: {type(self.model)}")
            
            # Get expected feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.expected_features = list(self.model.feature_names_in_)
                print(f"Model expects {len(self.expected_features)} features:")
                print(f"Features: {self.expected_features}")
            else:
                print("Warning: Model doesn't have feature_names_in_ attribute")
                # Fallback to manual feature list based on your config
                self.expected_features = [
                    'user_first_engagement', 'user_pseudo_id', 'is_enable', 
                    'bounced', 'country_name', 'device_os', 'device_lang',
                    'cnt_user_engagement', 'cnt_level_start_quickplay', 
                    'cnt_level_end_quickplay', 'cnt_level_complete_quickplay',
                    'cnt_level_reset_quickplay', 'cnt_post_score', 
                    'cnt_spend_virtual_currency', 'cnt_ad_reward',
                    'cnt_challenge_a_friend', 'cnt_completed_5_levels', 
                    'cnt_use_extra_steps'
                ]
            
            self.ready = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.ready = False
            return False

    def _json_to_dataframe(self, json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert JSON data to pandas DataFrame with proper column ordering.
        
        Args:
            json_data: List of dictionaries with feature data
            
        Returns:
            pandas DataFrame with features in correct order
        """
        # Create DataFrame from JSON
        df = pd.DataFrame(json_data)
        
        # Ensure all expected features are present
        missing_features = []
        for feature in self.expected_features:
            if feature not in df.columns:
                missing_features.append(feature)
                df[feature] = None  # Add missing columns with None
        
        if missing_features:
            print(f"Warning: Missing features filled with None: {missing_features}")
        
        # Reorder columns to match model expectation
        df = df[self.expected_features]
        
        print(f"Created DataFrame with shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        
        return df

    def _parse_input_data(self, request: InferenceRequest) -> List[Dict[str, Any]]:
        """
        Parse input data from MLServer request into JSON format.
        
        Supports multiple input formats:
        1. JSON objects in 'data' field
        2. Raw JSON strings  
        3. Individual feature inputs
        """
        if not request.inputs:
            raise ValueError("No inputs provided in request")
        
        input_data = request.inputs[0]
        
        # Handle MLServer TensorData wrapping
        data_values = input_data.data
        if hasattr(data_values, '__iter__') and not isinstance(data_values, str):
            # Convert TensorData to list if needed
            data_list = list(data_values)
        else:
            data_list = [data_values]
        
        print(f"Received data type: {type(data_values)}")
        print(f"Data list length: {len(data_list)}")
        print(f"First item type: {type(data_list[0]) if data_list else 'None'}")
        
        # Method 1: Direct JSON objects in data field
        if data_list and isinstance(data_list[0], dict):
            print("Received JSON objects directly")
            return data_list
        elif data_list and isinstance(data_list[0], str):
            # Method 2: JSON strings that need parsing
            import json
            try:
                parsed_data = []
                for item in data_list:
                    parsed_data.append(json.loads(item))
                print("Parsed JSON strings")
                return parsed_data
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string in input data")
        
        # Method 3: Feature values as flat list (reconstruct to dict)
        if hasattr(input_data, 'shape') and len(input_data.shape) >= 2:
            n_samples = input_data.shape[0] 
            n_features = input_data.shape[1]
            
            if n_features == len(self.expected_features):
                print(f"Reconstructing {n_samples} samples from flat data")
                samples = []
                
                for i in range(n_samples):
                    start_idx = i * n_features
                    end_idx = start_idx + n_features
                    sample_values = data_list[start_idx:end_idx]
                    
                    sample_dict = {}
                    for j, feature_name in enumerate(self.expected_features):
                        sample_dict[feature_name] = sample_values[j]
                    
                    samples.append(sample_dict)
                
                return samples
        
        raise ValueError(f"Unsupported input format. Data type: {type(data_values)}, Content: {data_list[:2] if len(data_list) > 0 else 'Empty'}")

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """
        Main prediction method that handles JSON to DataFrame conversion.
        """
        try:
            print(f"Received prediction request with {len(payload.inputs)} inputs")
            
            # Parse input data to JSON format
            json_data = self._parse_input_data(payload)
            print(f"Parsed {len(json_data)} samples from input")
            
            # Convert JSON to DataFrame
            df = self._json_to_dataframe(json_data)
            
            # Make prediction using sklearn model
            print("Making prediction with sklearn model...")
            predictions = self.model.predict(df)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions: {predictions}")
            
            # Try to get prediction probabilities if available
            prediction_probs = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    prediction_probs = self.model.predict_proba(df)
                    print(f"Prediction probabilities shape: {prediction_probs.shape}")
                    print(f"Prediction probabilities: {prediction_probs}")
                except Exception as e:
                    print(f"Could not get prediction probabilities: {e}")
            
            # Format response
            outputs = []
            
            # Add predictions
            outputs.append(
                ResponseOutput(
                    name="predictions",
                    shape=list(predictions.shape),
                    datatype="INT64",
                    data=predictions.tolist()
                )
            )
            
            # Add probabilities if available
            if prediction_probs is not None:
                outputs.append(
                    ResponseOutput(
                        name="probabilities", 
                        shape=list(prediction_probs.shape),
                        datatype="FP64",
                        data=prediction_probs.flatten().tolist()
                    )
                )
            
            response = InferenceResponse(
                model_name=self.name,
                model_version=self.version,
                outputs=outputs
            )
            
            print("Successfully created response")
            return response
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise e