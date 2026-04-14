const express = require("express");
const router = express.Router();

const { predictMatch } = require("../controllers/predictionController");

router.post("/", predictMatch);

module.exports = router;