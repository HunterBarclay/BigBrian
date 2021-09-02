using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Linq;

namespace BigBrian.v2 {

    public static class Rando {
        public static Random rand;

        static Rando() {
            rand = new Random(DateTime.Now.Millisecond);
        }

        public static double NextDouble() => rand.NextDouble();
    }

    public class OmegaTrainer {

        public const double DELTA_ADJUSTMENT = 5;
        public TestCase[] Data;
        public Network TargetNetwork;

        public double LastCost = double.NaN;

        public List<(double[][][] weightDelta, double[] biasDelta)> gradients = new List<(double[][][] weightDelta, double[] biasDelta)>();

        public OmegaTrainer(int[] structure, TestCase[] data) {
            TargetNetwork = new Network(structure);
            Data = data;
        }

        public void Test(bool printImprovement) {
            double cost = 0;
            foreach (var d in Data) {
                string o = $"    IN: [ {d.inputs[0]}";
                foreach (var a in d.inputs.Skip(1)) { o += $", {a}"; }
                o += $" ]\n    OUT: [ {d.outputs[0]}";
                foreach (var a in d.outputs.Skip(1)) { o += $", {a}"; }
                var actual = Calculate(d.inputs, d.outputs);
                o += $" ]\n    ACTUAL: [ {actual[0]}";
                foreach (var a in actual.Skip(1)) { o += $", {a}"; }
                o += $" ]";
                for (int i = 0; i < d.outputs.Length; i++) {
                    cost += Math.Pow(d.outputs[i] - actual[i], 2);
                }
                // Console.WriteLine(o);
            }
            if (printImprovement)
                Console.WriteLine($" Improvement -> {LastCost - cost}");
            LastCost = cost;
            // Console.WriteLine($"\n\n Cost -> {cost}\n\n");
            ApplyGradient();
        }

        public double[] Calculate(double[] inputs, double[] outputs) {
            var results = TargetNetwork.Calculate(inputs);

            var gradient = CalculateGradient(outputs, results);
            //for (int x = 0; x < gradient.weightDelta.Length; ++x) {
            //    for (int y = 0; y < gradient.weightDelta[x].Length; ++y) {
            //        for (int z = 0; z < gradient.weightDelta[x][y].Length; ++z) {
            //            Console.WriteLine($"Weight [{x}][{y}][{z}] -> {gradient.weightDelta[x][y][z]}");
            //        }
            //    }
            //    Console.WriteLine($"Bias [{x}] -> {gradient.biasDelta[x]}");
            //}
            gradients.Add(gradient);
            // ApplyGradient(gradient);

            return results;
        }

        public void ApplyGradient() {
            (double[][][] weightDelta, double[] biasDelta) gradient = gradients[0];

            for (int i = 1; i < gradients.Count; ++i) {
                for (int x = 0; x < gradient.weightDelta.Length; ++x) {
                    for (int y = 0; y < gradient.weightDelta[x].Length; ++y) {
                        for (int z = 0; z < gradient.weightDelta[x][y].Length; ++z) {
                            gradient.weightDelta[x][y][z] += gradients[i].weightDelta[x][y][z];
                        }
                    }
                    gradient.biasDelta[x] += gradients[i].biasDelta[x];
                }
            }

            for (int x = 0; x < gradient.weightDelta.Length; ++x) {
                for (int y = 0; y < gradient.weightDelta[x].Length; ++y) {
                    for (int z = 0; z < gradient.weightDelta[x][y].Length; ++z) {
                        TargetNetwork.Layers[x].Weights[y][z] -= (gradient.weightDelta[x][y][z] / gradients.Count) * DELTA_ADJUSTMENT;
                    }
                }
                TargetNetwork.Layers[x].Bias -= (gradient.biasDelta[x] / gradients.Count) * DELTA_ADJUSTMENT;
            }

            gradients.Clear();
        }

        private (double[][][] weightDelta, double[] biasDelta) CalculateGradient(double[] target, double[] actual) {
            double[][][] weightGradient = new double[TargetNetwork.Layers.Length][][];
            double[] biasGradient = new double[TargetNetwork.Layers.Length];
            for (int l = 0; l < TargetNetwork.Layers.Length; ++l) {
                weightGradient[l] = new double[TargetNetwork.Structure[l]][];
                for (int i = 0; i < TargetNetwork.Structure[l]; ++i) {
                    weightGradient[l][i] = new double[TargetNetwork.Structure[l + 1]];
                    for (int o = 0; o < TargetNetwork.Structure[l + 1]; ++o) {
                        double weightSum = 0;
                        if ((TargetNetwork.Layers.Length - 1) == l) {
                            weightSum = 2 * (actual[o] - target[o]) * Derivative(TargetNetwork.Layers.Length - 1, o, l, (i, o), false);
                        } else {
                            for (int resultNode = 0; resultNode < target.Length; ++resultNode) {
                                weightSum += 2 * (actual[resultNode] - target[resultNode]) * Derivative(TargetNetwork.Layers.Length - 1, resultNode, l, (i, o), false);
                            }
                        }

                        weightGradient[l][i][o] = weightSum;
                    }
                }

                double biasSum = 0;
                for (int resultNode = 0; resultNode < target.Length; ++resultNode) {
                    biasSum += 2 * (actual[resultNode] - target[resultNode]) * Derivative(TargetNetwork.Layers.Length - 1, resultNode, l, default, true);
                }
                biasGradient[l] = biasSum;
            }
            return (weightGradient, biasGradient);
        }

        public double Derivative(int currentLayer, int currentNode, int targetLayer, (int prev, int outp) targetWeight, bool getBias) {
            double deriv = TargetNetwork.Layers[currentLayer].ActivationDerivatives[currentNode];

            if (getBias) {
                if (currentLayer != targetLayer) {
                    double sum = 0;
                    for (int i = 0; i < TargetNetwork.Structure[currentLayer]; ++i) {
                        sum += TargetNetwork.Layers[currentLayer].Weights[i][currentNode]
                            * Derivative(currentLayer - 1, i, targetLayer, targetWeight, getBias);
                    }
                    deriv *= sum;
                }
            } else {
                if (currentLayer == targetLayer) {
                    deriv *= TargetNetwork.Layers[currentLayer].PreviousActivation[targetWeight.prev];
                } else if (currentLayer - targetLayer < 2) {
                    deriv *= TargetNetwork.Layers[currentLayer].Weights[targetWeight.outp][currentNode]
                        * Derivative(currentLayer - 1, targetWeight.outp, targetLayer, targetWeight, getBias);
                } else {
                    double sum = 0;
                    for (int i = 0; i < TargetNetwork.Structure[currentLayer]; ++i) {
                        sum += TargetNetwork.Layers[currentLayer].Weights[i][currentNode]
                            * Derivative(currentLayer - 1, i, targetLayer, targetWeight, getBias);
                    }
                    deriv *= sum;
                }
            }

            return deriv;
        }

        public class Network {
            public int[] Structure;
            public Layer[] Layers;

            public Network(int[] structure) {
                Structure = structure;
                Init();
            }

            private void Init() {
                Layers = new Layer[Structure.Length - 1];
                for (int l = 0; l < Structure.Length - 1; l++) {
                    Layers[l] = new Layer(new int[] { Structure[l], Structure[l + 1] }, Sigmoid, SigmoidDer);
                }
            }

            public double[] Calculate(double[] input) {
                for (int l = 0; l < Layers.Length; ++l) {
                    input = Layers[l].Calculate(input);
                }
                return input;
            }

            public static double Sigmoid(double a) => 1 / (1 + Math.Pow(Math.E, -a));
            public static double SigmoidDer(double a) => Sigmoid(a) * (1 - Sigmoid(a));
        }

        public class Layer {
            private int[] _structure;
            public double Bias;
            public double[][] Weights; // [input node][output node]
            public double[] PreviousActivation;
            public double[] BeforeActivation;
            private double[] _afterActivation;
            private Func<double, double> _activation;
            private Func<double, double> _activationDeriv;

            public double[] ActivationDerivatives;

            private int outputNodeCount => _structure[1];
            private int inputNodeCount => _structure[0];

            public Layer(int[] structure, Func<double, double> activation, Func<double, double> activationDeriv) {
                _structure = structure;
                _activation = activation;
                _activationDeriv = activationDeriv;
                Randomize(DateTime.Now.Millisecond);
            }

            private void Randomize(int seed) {
                Random rand = new Random(seed);

                Weights = new double[inputNodeCount][];
                for (int inputNode = 0; inputNode < inputNodeCount; ++inputNode) {
                    Weights[inputNode] = new double[outputNodeCount];
                    for (int outputNode = 0; outputNode < outputNodeCount; ++outputNode) {
                        Weights[inputNode][outputNode] = (Rando.NextDouble() * 2) - 1;
                    }
                }
                Bias = (Rando.NextDouble() * 1) - 0.5;
            }

            public double[] Calculate(double[] input) {
                PreviousActivation = input;
                BeforeActivation = new double[outputNodeCount];
                _afterActivation = new double[outputNodeCount];
                ActivationDerivatives = new double[outputNodeCount];

                for (int o = 0; o < outputNodeCount; ++o) {
                    double productSum = Bias;
                    for (int i = 0; i < inputNodeCount; ++i) {
                        productSum += Weights[i][o] * input[i];
                    }
                    BeforeActivation[o] = productSum;
                    ActivationDerivatives[o] = _activationDeriv(BeforeActivation[o]);
                    _afterActivation[o] = _activation(BeforeActivation[o]);
                }

                return _afterActivation;
            }
        }
    }

    public struct TestCase {
        public double[] inputs;
        public double[] outputs;
    }
}
