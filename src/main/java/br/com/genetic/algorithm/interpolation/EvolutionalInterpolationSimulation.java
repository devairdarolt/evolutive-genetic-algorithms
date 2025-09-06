package br.com.genetic.algorithm.interpolation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import lombok.ToString;

class Params {
	static final Random RNG = new Random(42);

	// ==== Dados (configuráveis) ====
	static final int NUM_PONTOS = 60;
	static final double X_MIN = -3, X_MAX = 3;
	static final double NOISE_STD = 0.20;

	// ==== Modelo base ====
	static final int GRAU_INICIAL = 0;
	static final double COEF_MIN = -5, COEF_MAX = 5;
	static final double DELTA_MIN = -1, DELTA_MAX = 1;

	// ==== Telômeros ====
	static final int TEL_X_INIT = 1000;
	static final int TEL_Y_INIT = 1000;

	// ==== População dinâmica (steady-state) ====
	static final int POP_INIT = 2; // começa com 2
	static final int POP_MAX = 10; // teto
	static final int BIRTHS_PER_GEN = 2; // filhos gerados por geração
	static final double IMIGRANT_RATE = 0.10; // prob/geração de adicionar 1 imigrante
	static final double RANDOM_PICK_RATE = 0.10; // chance de escolher pai/mãe aleatório (anti-congelamento)

	// Mantém proporção mínima de XY
	static final double FRAQ_INICIAL_XY = 0.30;
	static final double PROB_FILHO_XY = 0.25;

	static final int TORNEIO_XX = 3, TORNEIO_XY = 3;
	static final int GERACOES = 120000;

	// ==== Mutação ====
	static final double MUT_RATE_COEF = 0.10;
	static final double MUT_SIGMA_X = 0.15;
	static final double MUT_SIGMA_Y = 0.10;

	// ==== Tendência Y→X ====
	static final double ETA_TREND = 0.50;

	// ==== Mutação estrutural de X ====
	static final double PROB_MUT_GRAVE = 0.60;
	static final int ADD_COEF_MIN = 1;
	static final int ADD_COEF_MAX = 2;
	static final double NOVO_COEF_RANGE = 0.8;

	// ==== Salvaguarda contra colapso de grau ====
	static final int L_UTIL_MIN = 2; // nunca deixar útil < 2
	static final double PROB_ELITE_REJUVENATE = 1.0; // 100%: após corte útil, tenta mutação grave já na elite

	static double uniform(double a, double b) {
		return a + (b - a) * RNG.nextDouble();
	}

	static double gauss(double s) {
		return RNG.nextGaussian() * s;
	}

	static double clamp(double v, double lo, double hi) {
		return Math.max(lo, Math.min(hi, v));
	}
}

/* ===================== CROMOSSOMOS ===================== */
abstract class Cromossomo {
	int telomeros;

	abstract Cromossomo copiaEnvelhecendo();
}

@ToString
class X extends Cromossomo {

	double[] gene;

	boolean lastTrimmedUseful = false;

	X(double[] gene, int telomeros) {
		this.gene = gene;
		this.telomeros = telomeros;
	}

	int lenUtil() {
		return Math.max(0, gene.length - telomeros);
	}

	double[] getUtil() {
		return Arrays.copyOf(gene, lenUtil());
	}

	double avaliar(double x) {
		double y = 0.0, pot = 1.0;
		for (double c : gene) {
			y += c * pot;
			pot *= x;
		}
		return y;
	}

	void mutarLeve() {
		int L = lenUtil();
		for (int i = 0; i < L; i++) {
			if (Params.RNG.nextDouble() < Params.MUT_RATE_COEF) {
				gene[i] += Params.gauss(Params.MUT_SIGMA_X);
				gene[i] = Params.clamp(gene[i], Params.COEF_MIN, Params.COEF_MAX);
			}
		}
	}

	void talvezMutacaoGrave() {
		// dispara em 2 casos:
		// (a) houve truncamento útil recente; (b) útil caiu abaixo do mínimo (salvaguarda)
		boolean precisa = lastTrimmedUseful || lenUtil() < Params.L_UTIL_MIN;
		if (!precisa)
			return;

		if (!lastTrimmedUseful && lenUtil() < Params.L_UTIL_MIN) {
			// forçar a ocorrer mesmo se a moeda não ajudar
		} else if (Params.RNG.nextDouble() > Params.PROB_MUT_GRAVE) {
			return;
		}

		int add = Params.ADD_COEF_MIN + Params.RNG.nextInt(Params.ADD_COEF_MAX - Params.ADD_COEF_MIN + 1);
		int Lnovo = gene.length + add;
		double[] g2 = Arrays.copyOf(gene, Lnovo);
		for (int i = gene.length; i < Lnovo; i++) {
			g2[i] = Params.uniform(-Params.NOVO_COEF_RANGE, Params.NOVO_COEF_RANGE);
		}
		gene = g2;
		telomeros = Params.TEL_X_INIT; // reset de telômeros ao adicionar grau
		lastTrimmedUseful = false;
	}

	@Override
	X copiaEnvelhecendo() {
		double[] c = Arrays.copyOf(gene, gene.length);
		int t = telomeros;
		boolean trimmedUseful = false;

		if (t > 0) {
			c = Arrays.copyOf(c, c.length - 1);
			t -= 1;
		} else {
			if (c.length > 1) {
				c = Arrays.copyOf(c, c.length - 1);
				trimmedUseful = true;
			}
		}
		if (c.length == 0)
			c = new double[] { 0.0 };

		X nx = new X(c, t);
		nx.lastTrimmedUseful = trimmedUseful;
		// salvaguarda imediata: se útil caiu demais, já sinaliza (mutação grave virá no pós-processo)
		return nx;
	}

	X copiaSemEnvelhecer() {
		X nx = new X(Arrays.copyOf(gene, gene.length), telomeros);
		nx.lastTrimmedUseful = this.lastTrimmedUseful;
		return nx;
	}

	static X novoAleatorio() {
		int Lutil = Params.GRAU_INICIAL + 1;
		int L = Lutil + Params.TEL_X_INIT;
		double[] c = new double[L];
		for (int i = 0; i < Lutil; i++)
			c[i] = Params.uniform(Params.COEF_MIN, Params.COEF_MAX);
		for (int i = Lutil; i < L; i++)
			c[i] = 0.0;
		return new X(c, Params.TEL_X_INIT);
	}
}

@ToString
class Y extends Cromossomo {
	double[] gene;

	Y(double[] delta, int telomeros) {
		this.gene = delta;
		this.telomeros = telomeros;
	}

	int lenUtil() {
		return Math.max(0, gene.length - telomeros);
	}

	void mutarLeve() {
        int L = lenUtil();
        for (int i = 0; i < L; i++) {
            if (Params.RNG.nextDouble() < Params.MUT_RATE_COEF) {
                gene[i] += Params.gauss(Params.MUT_SIGMA_Y);
                gene[i] = Params.clamp(gene[i], Params.DELTA_MIN, Params.DELTA_MAX);
            }
        }
    }

	@Override
    Y copiaEnvelhecendo() {
        double[] d = Arrays.copyOf(gene, gene.length);
        int t = telomeros;

        if (t > 0) { d = Arrays.copyOf(d, d.length - 1); t -= 1; }
		else {
			if (d.length > 1)
				d = Arrays.copyOf(d, d.length - 1);
		}
		if (d.length == 0)
			d = new double[] { 0.0 };
		return new Y(d, t);
	}

	Y copiaSemEnvelhecer() {
		return new Y(Arrays.copyOf(gene, gene.length), telomeros);
	}

	static Y novoAleatorio() {
		int Lutil = Params.GRAU_INICIAL + 1;
		int L = Lutil + Params.TEL_Y_INIT;
		double[] d = new double[L];
		for (int i = 0; i < Lutil; i++)
			d[i] = Params.uniform(-0.5, 0.5);
		for (int i = Lutil; i < L; i++)
			d[i] = 0.0;
		return new Y(d, Params.TEL_Y_INIT);
	}
}

/* ===================== INDIVÍDUOS ===================== */
interface Individuo {

	X fenotipoX();

	Individuo copiaEnvelhecendo();

	Individuo copiaSemEnvelhecer();
}

@ToString
class Feminino implements Individuo { // XX
	final X x1, x2;

	Feminino(X x1, X x2) {
		this.x1 = x1;
		this.x2 = x2;
	}

	public X fenotipoX() {
		int L = Math.max(x1.gene.length, x2.gene.length);
		double[] m = new double[L];
		for (int i = 0; i < L; i++) {
			double a = valorIndexado(x1, i);
			double b = valorIndexado(x2, i);
			m[i] = 0.5 * (a + b);
		}
		int t = Math.min(x1.telomeros + Math.max(0, L - x1.gene.length),
				x2.telomeros + Math.max(0, L - x2.gene.length));
		t = Math.max(0, Math.min(t, L));
		return new X(m, t);
	}

	private double valorIndexado(X x, int i) {
		if (i >= x.gene.length)
			return 0.0;
		int util = x.lenUtil();
		return (i < util) ? x.gene[i] : 0.0;
	}

	public Individuo copiaEnvelhecendo() {
		return new Feminino(x1.copiaEnvelhecendo(), x2.copiaEnvelhecendo());
	}

	public Individuo copiaSemEnvelhecer() {
		return new Feminino(x1.copiaSemEnvelhecer(), x2.copiaSemEnvelhecer());
	}

	X getX1() {
		return x1;
	}

	X getX2() {
		return x2;
	}
}

@ToString
class Masculino implements Individuo { // XY
	final X x;
	final Y y;

	Masculino(X x, Y y) {
		this.x = x;
		this.y = y;
	}

	public X fenotipoX() {
		return x;
	}

	public Individuo copiaEnvelhecendo() {
		return new Masculino(x.copiaEnvelhecendo(), y.copiaEnvelhecendo());
	}

	public Individuo copiaSemEnvelhecer() {
		return new Masculino(x.copiaSemEnvelhecer(), y.copiaSemEnvelhecer());
	}

	X getX() {
		return x;
	}

	Y getY() {
		return y;
	}
}

/* ===================== POPULAÇÃO & AVALIAÇÃO ===================== */
class Populacao {
	final List<Feminino> femininos = new ArrayList<>();
	final List<Masculino> masculinos = new ArrayList<>();

	final Map<Feminino, Double> rankFeminino = new LinkedHashMap<>();
	final Map<Masculino, Double> rankMasculino = new LinkedHashMap<>();

	int tamanho() {
		return femininos.size() + masculinos.size();
	}

	void adicionar(Feminino f) {
		femininos.add(f);
	}

	void adicionar(Masculino m) {
		masculinos.add(m);
	}

	void removerAleatorio() {
		// sorteia sexo primeiro, mas garante não zerar
		boolean removeFem = Params.RNG.nextBoolean();
		if (removeFem && femininos.size() > 1) {
			femininos.remove(Params.RNG.nextInt(femininos.size()));
		} else if (masculinos.size() > 1) {
			masculinos.remove(Params.RNG.nextInt(masculinos.size()));
		} else if (femininos.size() > 1) {
			femininos.remove(Params.RNG.nextInt(femininos.size()));
		} else if (masculinos.size() > 1) {
			masculinos.remove(Params.RNG.nextInt(masculinos.size()));
		}
		// não remove se isso mataria o último de um sexo e o outro também já está com 1
	}

	void limparRanks() {
		rankFeminino.clear();
		rankMasculino.clear();
	}

	void ranquear(java.util.function.ToDoubleFunction<Individuo> fit) {
		limparRanks();
		femininos.stream().sorted((a, b) -> Double.compare(fit.applyAsDouble(b), fit.applyAsDouble(a)))
				.forEach(f -> rankFeminino.put(f, fit.applyAsDouble(f)));
		masculinos.stream().sorted((a, b) -> Double.compare(fit.applyAsDouble(b), fit.applyAsDouble(a)))
				.forEach(m -> rankMasculino.put(m, fit.applyAsDouble(m)));
	}
}

class Avaliacao {
	
	static double fitness(Individuo ind, List<double[]> dadosXY) {
	    X fx = ind.fenotipoX();
	    double s = 0.0;
	    for (double[] par : dadosXY) {
	        double x = par[0], y = par[1];
	        double yp = fx.avaliar(x);

	        // guarda-corpo numérico
	        if (!Double.isFinite(yp)) return 0.0;

	        double e = y - yp;
	        if (!Double.isFinite(e)) return 0.0;

	        s += e * e; // ou: Math.abs(e) se preferir soma absoluta
	        if (!Double.isFinite(s)) return 0.0; // overflow => fitness 0
	    }

	    double mse = s / dadosXY.size();
	    if (!Double.isFinite(mse)) return 0.0;
	    return 1.0 / (1.0 + mse);
	}

}

/* ===================== FÁBRICA / DADOS ===================== */
class Fabrica {
	static Feminino novoFeminino() {
		return new Feminino(X.novoAleatorio(), X.novoAleatorio());
	}

	static Masculino novoMasculino() {
		return new Masculino(X.novoAleatorio(), Y.novoAleatorio());
	}

	static List<double[]> gerarDados(double[] coef, int nPts, double xMin, double xMax, double noiseStd) {
		List<double[]> ds = new ArrayList<>();
		for (int i = 0; i < nPts; i++) {
			double x = xMin + (xMax - xMin) * i / (nPts - 1.0);
			double y = avaliarPolinomio(coef, x) + Params.gauss(noiseStd);
			ds.add(new double[] { x, y });
		}
		return ds;
	}

	static double avaliarPolinomio(double[] coef, double x) {
		double y = 0.0, pot = 1.0;
		for (double c : coef) {
			y += c * pot;
			pot *= x;
		}
		return y;
	}
}

/* ===================== OPERADORES ===================== */
class Operadores {

	// torneios por sexo com jitter de aleatoriedade
	static Feminino torneioF(Populacao pop, List<double[]> dados) {
		if (pop.femininos.isEmpty())
			return Fabrica.novoFeminino();
		if (Params.RNG.nextDouble() < Params.RANDOM_PICK_RATE)
			return pop.femininos.get(Params.RNG.nextInt(pop.femininos.size()));

		Feminino best = null;
		double bf = -1;
		int k = Math.min(Params.TORNEIO_XX, Math.max(1, pop.femininos.size()));
		for (int i = 0; i < k; i++) {
			Feminino cand = pop.femininos.get(Params.RNG.nextInt(pop.femininos.size()));
			double f = Avaliacao.fitness(cand, dados);
			if (f > bf) {
				bf = f;
				best = cand;
			}
		}
		return (best != null) ? best : Fabrica.novoFeminino();
	}

	static Masculino torneioM(Populacao pop, List<double[]> dados) {
		if (pop.masculinos.isEmpty())
			return Fabrica.novoMasculino();
		if (Params.RNG.nextDouble() < Params.RANDOM_PICK_RATE)
			return pop.masculinos.get(Params.RNG.nextInt(pop.masculinos.size()));

		Masculino best = null;
		double bf = -1;
		int k = Math.min(Params.TORNEIO_XY, Math.max(1, pop.masculinos.size()));
		for (int i = 0; i < k; i++) {
			Masculino cand = pop.masculinos.get(Params.RNG.nextInt(pop.masculinos.size()));
			double f = Avaliacao.fitness(cand, dados);
			if (f > bf) {
				bf = f;
				best = cand;
			}
		}
		return (best != null) ? best : Fabrica.novoMasculino();
	}

	static void aplicarTendencia(X x, Y y) {
		int L = Math.min(x.lenUtil(), y.lenUtil());
		for (int i = 0; i < L; i++) {
			x.gene[i] = Params.clamp(x.gene[i] + Params.ETA_TREND * y.gene[i], Params.COEF_MIN, Params.COEF_MAX);
		}
	}

	static void mutarXCompleto(X x) {
		x.mutarLeve();
		x.talvezMutacaoGrave();
	}

	static void mutarY(Y y) {
		y.mutarLeve();
	}

	// rejuvenescimento pós-elitismo: se houve corte útil, tenta mutação grave já
	static void rejuvenescerSeCortou(Individuo ind) {
		if (Params.RNG.nextDouble() > Params.PROB_ELITE_REJUVENATE)
			return;
		if (ind instanceof Feminino) {
			Feminino f = (Feminino) ind;
			f.getX1().talvezMutacaoGrave();
			f.getX2().talvezMutacaoGrave();
		} else {
			Masculino m = (Masculino) ind;
			m.getX().talvezMutacaoGrave();
		}
	}

	// reprodução steady-state: gera 1 filho
	static Individuo reproduzir(Feminino mae, Masculino pai) {
		boolean nasceXY = Params.RNG.nextDouble() < Params.PROB_FILHO_XY;

		X xMae = (Params.RNG.nextBoolean() ? mae.getX1() : mae.getX2()).copiaEnvelhecendo();

		if (nasceXY) {
			Y yPai = pai.getY().copiaEnvelhecendo();
			X xFilho = xMae.copiaSemEnvelhecer();
			aplicarTendencia(xFilho, yPai);
			mutarXCompleto(xFilho);
			mutarY(yPai);
			return new Masculino(xFilho, yPai);
		} else {
			X xPai = pai.getX().copiaEnvelhecendo();
			mutarXCompleto(xMae);
			mutarXCompleto(xPai);
			return new Feminino(xMae, xPai);
		}
	}
}

/* ===================== SIMULAÇÃO (steady-state e população dinâmica) ===================== */
public class EvolutionalInterpolationSimulation {

	public static void run(double[] polinomio) {
		List<double[]> dados = Fabrica.gerarDados(polinomio, Params.NUM_PONTOS, Params.X_MIN, Params.X_MAX, Params.NOISE_STD);
		Populacao pop = inicializarPopulacaoMinima(); // 2 indivíduos

		Individuo melhorSnapshot = null;
		double bestFitEver = -1;

		for (int g = 1; g <= Params.GERACOES; g++) {
			// ranking atual
			pop.ranquear(ind -> Avaliacao.fitness(ind, dados));

			Feminino eliteF = pop.rankFeminino.keySet().stream().findFirst().orElse(null);
			Masculino eliteM = pop.rankMasculino.keySet().stream().findFirst().orElse(null);

			// snapshot do melhor
			Individuo bestGen = escolherMelhorDaGeracao(dados, eliteF, eliteM);
			double bestFit = (bestGen != null) ? Avaliacao.fitness(bestGen, dados) : -1;
			if (bestGen != null && bestFit > bestFitEver) {
				bestFitEver = bestFit;
				melhorSnapshot = bestGen.copiaSemEnvelhecer();
			}

			// elites "envelhecem" um pouco (cópias com envelhecimento) e rejuvenescem se cortaram útil
			// (opcional: manter “fantasmas” de elite injetados no pool para não perder qualidade)
			if (eliteF != null) {
				Feminino ef = (Feminino) eliteF.copiaEnvelhecendo();
				Operadores.rejuvenescerSeCortou(ef);
				inserirEstocastico(pop, ef);
			}
			if (eliteM != null) {
				Masculino em = (Masculino) eliteM.copiaEnvelhecendo();
				Operadores.rejuvenescerSeCortou(em);
				inserirEstocastico(pop, em);
			}

			// nascimentos por geração
			int births = Params.BIRTHS_PER_GEN;
			int alvoXYMin = Math.max(1, (int) Math.round(Params.FRAQ_INICIAL_XY * Math.max(Params.POP_INIT, pop.tamanho())));

			for (int i = 0; i < births; i++) {
				Feminino mae = Operadores.torneioF(pop, dados);
				Masculino pai = Operadores.torneioM(pop, dados);

				Individuo filho = Operadores.reproduzir(mae, pai);
				// reforço para não rarear XY
				if (contarXY(pop) < alvoXYMin && !(filho instanceof Masculino)) {
					// tenta gerar um XY extra
					Individuo f2 = Operadores.reproduzir(mae, pai);
					if (f2 instanceof Masculino)
						inserirEstocastico(pop, f2);
				}
				inserirEstocastico(pop, filho);
			}

			// imigração estocástica de diversidade
			if (Params.RNG.nextDouble() < Params.IMIGRANT_RATE) {
				if (Params.RNG.nextBoolean())
					inserirEstocastico(pop, Fabrica.novoFeminino());
				else
					inserirEstocastico(pop, Fabrica.novoMasculino());
			}

			// log
			if (g == 1 || g % 10 == 0 || g == Params.GERACOES) {
				print(pop, g, bestGen, bestFit);
			}
		}

		// sanity check final
		double fitFinal = Avaliacao.fitness(melhorSnapshot, Fabrica.gerarDados(
				polinomio, Params.NUM_PONTOS, Params.X_MIN, Params.X_MAX, Params.NOISE_STD));
		System.out.println("\n=== Melhor (snapshot) de toda a simulação ===");
		System.out.printf(Locale.US, "fitness=%.6f | coefs=%s%n",
				fitFinal, fmt(melhorSnapshot.fenotipoX().getUtil()));
	}

	public static void print(Populacao pop, int g, Individuo bestGen, double bestFit) {

		double[] bestuUtilGenes = bestGen.fenotipoX().getUtil();
		List<String> genes = Arrays.stream(bestuUtilGenes)
				.mapToObj(c -> String.format(Locale.US, "%.3f", c))
				.collect(Collectors.toList());

		System.out.printf(Locale.US,
				"G%3d | best=%.5f | XX=%d XY=%d | coefs(best)=%s%n",
				g, bestFit,
				pop.femininos.size(), pop.masculinos.size(),
				(bestGen != null ? genes.toString() : "n/a"));
	}

	// ===== helpers =====

	static Populacao inicializarPopulacaoMinima() {
		Populacao pop = new Populacao();
		// garante 1 XX + 1 XY
		pop.adicionar(Fabrica.novoFeminino());
		pop.adicionar(Fabrica.novoMasculino());
		return pop;
	}

	static void inserirEstocastico(Populacao pop, Individuo ind) {
		// se abaixo do teto, só adiciona; se no teto, mata aleatoriamente e entra o novo
		if (pop.tamanho() < Params.POP_MAX) {
			if (ind instanceof Feminino)
				pop.adicionar((Feminino) ind);
			else
				pop.adicionar((Masculino) ind);
		} else {
			// morte aleatória, mas tenta não zerar algum sexo
			pop.removerAleatorio();
			if (ind instanceof Feminino)
				pop.adicionar((Feminino) ind);
			else
				pop.adicionar((Masculino) ind);
		}
	}

	static int contarXY(Populacao pop) {
		return pop.masculinos.size();
	}

	static Individuo escolherMelhorDaGeracao(List<double[]> dados, Feminino eliteF, Masculino eliteM) {
		double fitF = (eliteF != null) ? Avaliacao.fitness(eliteF, dados) : -1;
		double fitM = (eliteM != null) ? Avaliacao.fitness(eliteM, dados) : -1;
		return (fitF >= fitM ? eliteF : eliteM);
	}

	static String fmt(double[] c) {
		return IntStream.range(0, c.length)
				.mapToObj(i -> String.format(Locale.US, "a%d=%.3f", i, c[i]))
				.collect(Collectors.joining(", "));
	}

}
