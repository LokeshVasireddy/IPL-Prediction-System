const express = require("express");
const app = express();

const routes = require("./routes");
const { logger } = require("./middlewares/loggerMiddleware");
const { errorHandler } = require("./middlewares/errorMiddleware");

app.use(express.json());
app.use(logger);

app.use("/api", routes);

app.use(errorHandler);

module.exports = app;