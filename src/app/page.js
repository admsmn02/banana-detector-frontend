'use client';

import React, { useState } from "react";
import * as ort from "onnxruntime-web";

export default function DetectPage() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    try {
      const modelPath = "/models/banana_detector.onnx";
      const session = await ort.InferenceSession.create(modelPath);

      const tensor = await preprocessImage(file);
      const feeds = { input: tensor };
      const results = await session.run(feeds);
      const probability = results.output.data[0];

      setPrediction(probability < 0.5 ? "🍌 Banana Detected!" : "❌ No Banana Detected.");
    } catch (error) {
      console.error("Error during prediction:", error);
      setPrediction("Error during prediction.");
    }
    setLoading(false);
  };

  const preprocessImage = async (file) => {
    const imageBitmap = await createImageBitmap(file);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(imageBitmap, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224);
    const { data } = imageData;

    const normalizedData = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < data.length; i += 4) {
      normalizedData[i / 4] = (data[i] / 255 - 0.5) / 0.5; // Red
      normalizedData[i / 4 + 224 * 224] = (data[i + 1] / 255 - 0.5) / 0.5; // Green
      normalizedData[i / 4 + 2 * 224 * 224] = (data[i + 2] / 255 - 0.5) / 0.5; // Blue
    }

    return new ort.Tensor("float32", normalizedData, [1, 3, 224, 224]);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-r from-yellow-300 via-yellow-400 to-yellow-500 p-6">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-4xl font-extrabold text-yellow-600 text-center mb-6">
          Banana Detector 🍌
        </h1>
        <p className="text-gray-700 text-center mb-4">
          Upload an image, and let the detector tell you if there’s a banana!
          The detector is still struggling to find the banana if there are too many. Let's be kind to it. 🤖
        </p>

        <div className="flex flex-col items-center">
          <label
            htmlFor="file-upload"
            className="cursor-pointer bg-yellow-500 text-white font-semibold py-2 px-4 rounded-full shadow-md hover:bg-yellow-600 focus:outline-none focus:ring-2 focus:ring-yellow-400"
          >
            Upload Image
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
        </div>

        {loading && (
          <p className="text-center text-lg text-yellow-700 font-semibold mt-6">
            Processing your image...
          </p>
        )}

        {prediction && (
          <div className="mt-6 bg-gray-100 rounded-xl p-4 shadow-md text-center">
            <h2 className="text-2xl font-bold text-gray-800">{prediction}</h2>
          </div>
        )}
      </div>
    </div>
  );
}
