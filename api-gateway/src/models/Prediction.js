import mongoose from "mongoose";

const PredictionSchema = new mongoose.Schema({
  team_a: String,
  team_b: String,
  prediction: String,
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.model("Prediction", PredictionSchema);