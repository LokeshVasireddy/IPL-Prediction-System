import express from "express";
import dotenv from "dotenv";
import { connectDB } from "./config/db.js";
import predictRoutes from "./routes/predict.js";

dotenv.config();

const app = express();
app.use(express.json());

connectDB();

app.use("/api", predictRoutes);

app.listen(3000, () => console.log("Server running on port 3000"));