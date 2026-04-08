import axios from "axios";

export const getPrediction = async (data) => {
  const response = await axios.post(
    "http://ml-service:5000/predict",
    data
  );
  return response.data;
};