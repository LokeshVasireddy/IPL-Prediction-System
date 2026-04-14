const mlService = require("./mlService");

exports.getPrediction = async (input) => {
  const prediction = await mlService.predict(input);

  return {
    winProbability: prediction.winProb,
    predictedWinner: prediction.team
  };
};