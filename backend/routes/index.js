const express = require("express");
const router = express.Router();

const matchRoutes = require("./matchRoutes");
const predictionRoutes = require("./predictionRoutes");
const healthRoutes = require("./healthRoutes");

router.use("/matches", matchRoutes);
router.use("/predict", predictionRoutes);
router.use("/health", healthRoutes);

module.exports = router;