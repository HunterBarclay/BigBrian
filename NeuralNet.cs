using System;
using System.Collections.Generic;

/*
Cost = (activation - desired output)^2
dCost/dWeight = (activation of prev layer) (deriv of activation func of value before activation) (2) (activation - desired output)
dCost/dBias = (deriv of activation func of value before activation) (2) (activation - desired output)

TODO: Is there a bias per layer or bias per node??????

*/

namespace BigBrian {
    public class NeuralNet {

        public static double TrimMod = 0.003;

        public double Cost { get; set; }

        // public Func<double, double> Activation;
        // public Func<double, double> ActivationDer; // TODO
        
        private int[] structure;
        public int[] Structure { get => structure; }

        public Layer[] layers;

        public NeuralNet(int[] structure) {
            this.structure = structure;
        }

        public NeuralNet(int[] structure, int seed) {
            this.structure = structure;

            Gen(seed);
        }

        public NeuralNet(NeuralNet copyNet) { } // TODO

        public void Gen(int seed) {
            layers = new Layer[structure.Length - 1];
            for (int i = 1; i < structure.Length; i++) {
                layers[i - 1] = new Layer(structure[i], structure[i - 1], seed);
            }
        }

        public double[] Passthrough(double[] initLayer) {
            for (int i = 0; i < layers.Length; i++) {
                initLayer = layers[i].Passthrough(initLayer, i != layers.Length - 1); // Wtf fuck calculus
            }
            return initLayer;
        }

        public double[] Passthrough(double[] initLayer, double[] target) {
            var output = Passthrough(initLayer);
            CalculateCost(output, target);
            return output;
        }

        public void CalculateCost(double[] output, double[] target) {
            Cost = 0;
            for (int i = 0; i < output.Length; i++) {
                Cost += Math.Pow(output[i] - target[i], 2);
            }
            // Average??? idfk
        }

        public void CalculateDeltas(double[] output, double[] target) {
            for (int i = layers.Length - 1; i >= 0; i--) {
                if (i == layers.Length - 1)
                    layers[i].CalculateDeltas(output, target);
                else
                    layers[i].CalculateDeltas(layers[i+1]);

                layers[i].CacheDeltas();
            }
        }

        public void ApplyDeltas(bool clearCache) {
            foreach (var l in layers) {
                l.ApplyDeltas(clearCache);
            }
        }

        public class Layer {

            private Random random;

            private int size;
            public int Size { get => size; }
            private int weightsPerNode;
            public int WeightsPerNode { get => weightsPerNode; }

            public double[] biases;
            public double[] nodes;
            public double[][] weights;
            public double[] biasDeltas;
            public double[][] weightDeltas;
            public double[][] gammas;

            public List<DeltaCache> Deltas = new List<DeltaCache>();
            public TrainingDataDump trainingMeta;

            public Layer(int size, int weightsPerNode) {
                this.size = size;
                this.weightsPerNode = weightsPerNode;
            }

            public Layer(int size, int weightsPerNode, int seed) {
                this.size = size;
                this.weightsPerNode = weightsPerNode;

                Gen(seed);
            }
            
            public Layer(Layer copyLayer) {

            }

            public void Gen(int seed) {

                random = new Random(seed);

                nodes = new double[size];
                weights = new double[weightsPerNode][];
                for (int i = 0; i < weightsPerNode; i++) {
                    weights[i] = new double[size];
                    for (int j = 0; j < size; j++) {
                        weights[i][j] = (random.NextDouble() * 20) - 10; // -0.5 >= x < 0.5
                    }
                }
                biases = new double[size];
                for (int i = 0; i < biases.Length; i++) {
                    biases[i] = (random.NextDouble() * 20) - 10; // -0.5 >= x < 0.5
                }
            }

            public double[] Passthrough(double[] previousNodes, bool useBias = true) {
                trainingMeta.FedData = previousNodes;
                trainingMeta.BeforeActivation = new double[size];
                // trainingMeta.AfterActivation = new double[size];

                for (int i = 0; i < size; i++) {
                    nodes[i] = 0.0;
                    for (int j = 0; j < previousNodes.Length; j++) {
                        nodes[i] += weights[j][i] * previousNodes[j];
                    }
                    nodes[i] += useBias ? biases[i] : 0;
                    trainingMeta.BeforeActivation[i] = nodes[i];
                    nodes[i] = Activation(nodes[i]);
                    // trainingMeta.AfterActivation[i] = nodes[i];
                }
                return nodes;
            }

            public void CalculateDeltas(double[] output, double[] target) {

                // Weights
                weightDeltas = new double[weights.Length][];
                gammas = new double[weights.Length][];
                for (int i = 0; i < weightDeltas.Length; i++) {
                    weightDeltas[i] = new double[size];
                    gammas[i] = new double[size];
                    for (int j = 0; j < output.Length; j++) {
                        gammas[i][j] = (1.0 / (double)output.Length) * 2 * (output[j] - target[j]) * ActivationDer(trainingMeta.BeforeActivation[j]);
                        weightDeltas[i][j] = gammas[i][j] * trainingMeta.FedData[i];
                    }
                }

                // Bias
                biasDeltas = new double[biases.Length];
                for (int j = 0; j < output.Length; j++) {
                    biasDeltas[j] = (1.0 / (double)output.Length) * 2 * (output[j] - target[j]) * ActivationDer(trainingMeta.BeforeActivation[j]);
                }
                // for (int i = 0; i < weights.Length; i++) {
                //     for (int j = 0; j < output.Length; j++) {
                //         biasDelta += gammas[i][j];
                //     }
                // }
                // biasDelta /= output.Length; // Average out the bias delta cuz idfk what to do
            }

            public void CalculateDeltas(Layer foreLayer) {
                // Weights
                weightDeltas = new double[weights.Length][];
                gammas = new double[weights.Length][];
                for (int i = 0; i < weightDeltas.Length; i++) {
                    weightDeltas[i] = new double[size];
                    gammas[i] = new double[size];
                    for (int j = 0; j < size; j++) {
                        weightDeltas[i][j] = 0;
                        gammas[i][j] = 0;
                        for (int k = 0; k < foreLayer.size; k++) {
                            gammas[i][j] += foreLayer.gammas[j][k] * foreLayer.weights[j][k] * ActivationDer(trainingMeta.BeforeActivation[j]);
                            weightDeltas[i][j] += foreLayer.gammas[j][k] * foreLayer.weights[j][k]
                                * ActivationDer(trainingMeta.BeforeActivation[j]) * trainingMeta.FedData[i];
                        }
                        // weightDeltas[i][j] /= foreLayer.size;
                    }
                }

                // Bias
                biasDeltas = new double[biases.Length];
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < weights.Length; i++) {
                        double tempBiasDelta = 0.0;
                        for (int k = 0; k < foreLayer.size; k++) {
                            tempBiasDelta += (foreLayer.gammas[j][k])
                                * foreLayer.weights[j][k] * ActivationDer(trainingMeta.BeforeActivation[j]);
                        }
                        biasDeltas[j] = tempBiasDelta;// / foreLayer.size;
                    }
                }
                // biasDelta /= size; // Average out the bias delta cuz idfk what to do
            }

            public void CacheDeltas() {
                Deltas.Add(new DeltaCache() { biasDeltas = this.biasDeltas, weightsDelta = this.weightDeltas });
            }

            public void ApplyDeltas(bool clearCache) {
                // Get empty deltas;
                double[][] dW = weightDeltas;
                for (int x = 0; x < weightDeltas.Length; x++) {
                    for (int y = 0; y < weightDeltas[x].Length; y++) {
                        dW[x][y] = 0.0;
                    }
                }
                double[] dB = new double[size];

                // Sum
                foreach (var deltaSet in Deltas) {
                    for (int y = 0; y < size; y++) {
                            for (int x = 0; x < weightDeltas.Length; x++) {
                            dW[x][y] += deltaSet.weightsDelta[x][y];
                        }
                        dB[y] += deltaSet.biasDeltas[y];
                    }
                }

                // Average
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < weightDeltas.Length; y++) {
                        dW[y][x] /= Deltas.Count;
                    }
                    dB[x] /= Deltas.Count;
                }
                

                // Apply
                for (int x = 0; x < size; x++) {
                    for (int y = 0; y < weightDeltas.Length; y++) {
                        weights[y][x] -= dW[y][x] * NeuralNet.TrimMod; // TODO: Prob adjust trim mod based on reaction to cost after applying deltas
                    }
                    biases[x] -= dB[x] * NeuralNet.TrimMod;
                }

                if (clearCache)
                    Deltas.Clear();
            }

            public double Activation(double a) => Sigmoid(a);
            public double ActivationDer(double a) => SigmoidDer(a);

            public static double TanH(double a) => (Math.Pow(Math.E, a) - Math.Pow(Math.E, -a)) / (Math.Pow(Math.E, a) + Math.Pow(Math.E, -a));
            public static double TanHDer(double a) => 1 - Math.Pow(TanH(a), 2);

            public static double Sigmoid(double a) => 1 / (1 + Math.Pow(Math.E, -a));
            public static double SigmoidDer(double a) => Sigmoid(a) * (1 - Sigmoid(a));
        }

        public struct TrainingDataDump {
            public double[] FedData;
            public double[] BeforeActivation;
            // public double[] AfterActivation;
        }

        public struct DeltaCache {
            public double[] biasDeltas;
            public double[][] weightsDelta;
        }

    }
}