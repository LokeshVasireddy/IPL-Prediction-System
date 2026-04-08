const predictionService = require("../services/predictionService");

exports.predictMatch = async (req, res, next) => {
  try {
    const result = await predictionService.getPrediction(req.body);
    res.json({ success: true, data: result });
  } catch (error) {
    next(error);
  }
};