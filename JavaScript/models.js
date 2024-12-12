const tf = require("@tensorflow/tfjs-node");

// 1. Inflation Adjustment Model
function inflationAdjustedSalary(nominalSalary, inflationRate) {
  return nominalSalary * (1 + inflationRate / 100);
}

// 2. Compound Annual Growth Rate (CAGR)
function calculateCAGR(fv, pv, years) {
  return Math.pow(fv / pv, 1 / years) - 1;
}

// 3. Neural Network Regression (TensorFlow.js)
async function neuralNetworkRegression() {
  const xs = tf.tensor([1, 2, 3, 4]);
  const ys = tf.tensor([70, 85, 100, 130]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  await model.fit(xs, ys, { epochs: 500 });
  const prediction = model.predict(tf.tensor([5]));
  prediction.print();
}

// 4. Bayesian Regression (Not supported in JavaScript without advanced libraries)

// 5. GPT Example (Using OpenAI API)
const { Configuration, OpenAIApi } = require("openai");

async function gptExample() {
  const configuration = new Configuration({
    apiKey: "your-api-key",
  });
  const openai = new OpenAIApi(configuration);
  const response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt: "Predicting Full-Stack Developer salaries in 2025 involves",
    max_tokens: 50,
  });
  console.log(response.data.choices[0].text.trim());
}

// Running the Models
console.log("Inflation Adjusted Salary:", inflationAdjustedSalary(90000, 2.5));
console.log("CAGR:", (calculateCAGR(120000, 90000, 3) * 100).toFixed(2) + "%");
neuralNetworkRegression();
gptExample();
