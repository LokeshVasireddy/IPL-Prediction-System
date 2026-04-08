// Later: connect to Python model / API

exports.predict = async (input) => {
  // Dummy logic (replace with ML call)
  return {
    winProb: Math.random(),
    team: input.team_a
  };
};