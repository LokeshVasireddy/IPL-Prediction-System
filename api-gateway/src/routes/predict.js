import express from "express";
import Prediction from "../models/Prediction.js";
import { getPrediction } from "../services/mlService.js";

const router = express.Router();

router.post("/predict", async (req, res) => {
  try {
    const { team_a, team_b } = req.body;

    // 🔥 Step 1: Prepare features (later you add embeddings)
    const mlInput = { team_a, team_b };

    // 🔥 Step 2: Call ML service
    const result = await getPrediction(mlInput);

    // 🔥 Step 3: Save to DB
    const saved = await Prediction.create({
      team_a,
      team_b,
      prediction: result.prediction
    });

    // 🔥 Step 4: Return response
    res.json(saved);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;