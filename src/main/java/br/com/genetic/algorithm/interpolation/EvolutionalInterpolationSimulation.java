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


class Params {
	static final Random RNG = new Random(42);

	// dados alvo: y = 1 + 2x - 0.5x^2 + 0.3x^3 + ruído
	static final int NUM_PONTOS = 60;
	static final double X_MIN = -3, X_MAX = 3, NOISE_STD = 0.20;
	static final double[] ALVO = { 1.0, 2.0, -0.5, 0.3 };

	// modelo base
	static final int GRAU = 3; // grau útil inicial (=> GRAU+1 coef úteis)
	static final double COEF_MIN = -5, COEF_MAX = 5; // limites de coeficientes X
	static final double DELTA_MIN = -1, DELTA_MAX = 1; // limites de Y (tendência)

	// telômeros digitais
	static final int TEL_X_INIT = 6; // telômeros iniciais em X (zeros ao final)
	static final int TEL_Y_INIT = 6; // telômeros iniciais em Y (zeros ao final)

	// evolução
	static final int POP = 120;
	static final int GERACOES = 12000;
	static final double FRAQ_INICIAL_XY = 0.30;
	static final double PROB_FILHO_XY = 0.25; // probabilidade de nascer XY em XX×XY

	static final int TORNEIO_XX = 3, TORNEIO_XY = 3;
	static final int MIN_XY = Math.max(2, (int) Math.round(0.15 * POP)); // evitar extinção de XY

	// mutação
	static final double MUT_RATE_COEF = 0.10;
	static final double MUT_SIGMA_X = 0.15;
	static final double MUT_SIGMA_Y = 0.10;

	// tendência aplicada quando nasce XY (Y "empurra" X da mãe)
	static final double ETA_TREND = 0.50;

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

/*
 * ============================================================ ===================== CROMOSSOMOS =====================
 */
abstract class Cromossomo {
	
	int telomeros; // quantos genes finais ainda são telômeros silenciosos

	abstract Cromossomo copiaEnvelhecendo(); // "a cada cópia, remover o último telômero; se acabar, corta gene útil"
}

/**
 * Para Gene de X (que é um vetor de coeficientes do polinômio). * X é um cromossomo que contém coeficientes para um polinômio. * X tem telômeros que são zeros no final, representando a parte
 * silenciosa do gene.
 * 
 * * X é usado para representar o fenótipo de um indivíduo, avaliando o polinômio com os coeficientes fornecidos. * * X pode envelhecer, removendo telômeros ou truncando a parte útil do gene, o que
 * pode levar a uma redução no grau do polinômio.
 * 
 */
class X extends Cromossomo {

	double[] gene; // inclui genes úteis + telômeros (telômeros são zeros no final)

	X(double[] gene, int telomeros) {
		this.gene = gene;
		this.telomeros = telomeros;
	}

	// comprimento útil (exclui telômeros)
	int lenUtil() {
		return Math.max(0, gene.length - telomeros);
	}

	// avaliação do polinômio usando TODOS os coeficientes (telômeros são zeros -> neutros)
	double avaliar(double x) {
		double y = 0.0, pot = 1.0;
		for (double c : gene) {
			y += c * pot;
			pot *= x;
		}
		return y;
	}

	// muta apenas a parte útil (telômeros permanecem zeros inalterados)
	void mutar() {
		int L = lenUtil();
		for (int i = 0; i < L; i++) {
			if (Params.RNG.nextDouble() < Params.MUT_RATE_COEF) {
				gene[i] += Params.gauss(Params.MUT_SIGMA_X);
				gene[i] = Params.clamp(gene[i], Params.COEF_MIN, Params.COEF_MAX);
			}
		}
	}

	@Override
	X copiaEnvelhecendo() {
		// clona
		double[] c = Arrays.copyOf(gene, gene.length);
		int t = telomeros;

		// regra: a cada cópia, remove-se o último gene do array.
		if (t > 0) {
			// removendo um telômero (zero): não altera fenótipo, só reduz t e o tamanho do array
			c = Arrays.copyOf(c, c.length - 1);
			t -= 1;
		} else {
			// sem telômero: truncar gene útil (reduz grau): defeito potencial (variância disruptiva)
			if (c.length > 1)
				c = Arrays.copyOf(c, c.length - 1);
			// telômeros permanecem 0 (já não havia)
		}
		// nunca deixar sem nenhum coeficiente
		if (c.length == 0)
			c = new double[] { 0.0 };

		return new X(c, t);
	}

	X copiaSemEnvelhecer() {
		return new X(Arrays.copyOf(gene, gene.length), telomeros);
	}

	static X novoAleatorio() {
		int Lutil = Params.GRAU + 1;
		int L = Lutil + Params.TEL_X_INIT;
		double[] c = new double[L];
		for (int i = 0; i < Lutil; i++)
			c[i] = Params.uniform(Params.COEF_MIN, Params.COEF_MAX);
		for (int i = Lutil; i < L; i++)
			c[i] = 0.0; // telômeros silenciosos
		return new X(c, Params.TEL_X_INIT);
	}
}

/**
 * Para Gene de Y (que é um vetor de deltas para os coeficientes de X). * Y é um cromossomo que contém deltas (tendências) para os coeficientes de X. * Y tem telômeros que são zeros no final, como X.
 * * * Y é usado para guiar a evolução de X, aplicando tendências nos coeficientes de X.
 */
class Y extends Cromossomo {

	double[] gene; // deltas para coeficientes de X; telômeros são zeros no final

	Y(double[] delta, int telomeros) {
		this.gene = delta;
		this.telomeros = telomeros;
	}

	int lenUtil() {
		return Math.max(0, gene.length - telomeros);
	}

	void mutar() {
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

		if (t > 0) {
			d = Arrays.copyOf(d, d.length - 1);
			t -= 1;
		} else {
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
		int Lutil = Params.GRAU + 1;
		int L = Lutil + Params.TEL_Y_INIT;
		double[] d = new double[L];
		for (int i = 0; i < Lutil; i++)
			d[i] = Params.uniform(-0.5, 0.5);
		for (int i = Lutil; i < L; i++)
			d[i] = 0.0; // telômeros
		return new Y(d, Params.TEL_Y_INIT);
	}
}

/**
 * Interface para Individuos (Feminino e Masculino) na população. * Define métodos para calcular o fenótipo X, copiar o indivíduo envelhecendo ou sem envelhecer. * O fenótipo X é a representação do
 * indivíduo que será avaliada para fitness. * A cópia envelhecendo consome telômeros ou trunca a parte útil do gene, enquanto a cópia sem envelhecer mantém o estado atual.
 */
interface Individuo {

	X fenotipoX(); // como se calcula o X expressado para fitness

	Individuo copiaEnvelhecendo(); // copia "com divisão" (consome telômero / trunca útil)

	Individuo copiaSemEnvelhecer(); // copia "clonagem fria" (se precisar)
}

/**
 * Representa um indivíduo feminino (XX) na população. * Contém dois cromossomos X, representando os genes da mãe. * O fenótipo X é calculado como a média alinhada dos dois X, considerando telômeros.
 */
class Feminino implements Individuo { // XX

	final X x1, x2;

	Feminino(X x1, X x2) {
		this.x1 = x1;
		this.x2 = x2;
	}

	// fenótipo: média alinhada (se um X perde grau por truncamento, tratamos "faltantes" como 0)
	public X fenotipoX() {
		int L = Math.max(x1.gene.length, x2.gene.length);
		double[] m = new double[L];

		for (int i = 0; i < L; i++) {
			double a = valorIndexado(x1, i);
			double b = valorIndexado(x2, i);
			m[i] = 0.5 * (a + b);
		}
		// telômeros do fenótipo são os mínimos dos pais (parte silenciosa)
		int t = Math.min(x1.telomeros + Math.max(0, L - x1.gene.length),
				x2.telomeros + Math.max(0, L - x2.gene.length));
		t = Math.max(0, Math.min(t, L));
		return new X(m, t);
	}

	private double valorIndexado(X x, int i) {
		// índices >= lenUtil são telômeros (0); índices >= array.length também tratamos como 0
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

/**
 * Representa um indivíduo masculino (XY) na população. * Contém um cromossomo X e um cromossomo Y, representando os genes do pai e a tendência aplicada ao X. * O fenótipo X é calculado apenas com o
 * X, pois o Y não afeta diretamente o fenótipo, mas guia a evolução do X. * O Y é usado para aplicar tendências no X do filho XY, alinhando os coeficientes de X com os deltas de Y. * O fenótipo X é o
 * X do pai, que pode ser envelhecido ou não. * A cópia envelhecendo consome telômeros de X e Y, enquanto a cópia sem envelhecer mantém o estado atual.
 */
class Masculino implements Individuo { // XY

	final X x;
	final Y y;

	Masculino(X x, Y y) {
		this.x = x;
		this.y = y;
	}

	public X fenotipoX() {
		return x;
	} // Y só age na geração da prole

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

/**
 * Representa a população de indivíduos, contendo listas de femininos e masculinos.
 * 
 * <lo>
 * <li>Possui métodos para adicionar indivíduos, limpar a população, ranquear os indivíduos por fitness e calcular o tamanho da população.</li>
 * <li>Os rankings são armazenados em mapas separados para femininos e masculinos, permitindo acesso rápido aos melhores indivíduos de cada sexo.</li>
 * <li>O ranqueamento é feito com base em uma função de fitness, que avalia o desempenho dos indivíduos em relação aos dados fornecidos.</li>
 * <li>O tamanho da população é a soma dos indivíduos femininos e masculinos.</li>
 * <li>Os métodos de adição e limpeza permitem gerenciar a população de forma eficiente, facilitando a evolução dos indivíduos ao longo das gerações.</li>
 * <li>O ranqueamento é feito de forma a ordenar os indivíduos por fitness, permitindo identificar os melhores candidatos para reprodução.</li>
 * <li>Os rankings são armazenados em mapas para fácil acesso e comparação entre os indivíduos de cada sexo.</li>
 */
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

	void limpar() {
		femininos.clear();
		masculinos.clear();
	}

	void limparRanks() {
		rankFeminino.clear();
		rankMasculino.clear();
	}

	void ranquear(java.util.function.ToDoubleFunction<Individuo> fit) {
		limparRanks();
		femininos.stream()
				.sorted((a, b) -> Double.compare(fit.applyAsDouble(b), fit.applyAsDouble(a)))
				.forEach(f -> rankFeminino.put(f, fit.applyAsDouble(f)));
		masculinos.stream()
				.sorted((a, b) -> Double.compare(fit.applyAsDouble(b), fit.applyAsDouble(a)))
				.forEach(m -> rankMasculino.put(m, fit.applyAsDouble(m)));
	}
}

/**
 * Avaliação de fitness para indivíduos. <lo>
 * <li>Calcula o fitness de um indivíduo com base em dados XY fornecidos.</li>
 * <li>Utiliza o fenótipo X do indivíduo para avaliar a diferença entre os valores esperados (dados XY) e os valores calculados pelo polinômio representado por X.</li>
 * <li>O fitness é calculado como o inverso do erro quadrático médio (MSE) entre os valores esperados e os valores calculados.</li>
 * <li>Quanto menor o MSE, maior o fitness do indivíduo.</li>
 * <li>O fitness é normalizado para um valor entre 0 e 1, onde 1 representa o melhor ajuste possível.</li>
 * <li>Essa abordagem permite avaliar a qualidade dos indivíduos na população e direcionar a evolução para encontrar soluções melhores ao longo das gerações.</li>
 * <li>O fitness é usado para selecionar os melhores indivíduos para reprodução, garantindo que os descendentes herdem características vantajosas.</li>
 * <li>A avaliação é feita em relação a um conjunto de dados XY, que representa o comportamento esperado do polinômio alvo.</li> </lo>
 */
class Avaliacao {

	static double fitness(Individuo ind, List<double[]> dadosXY) {
		X fx = ind.fenotipoX();
		double s = 0.0;
		for (double[] par : dadosXY) {
			double x = par[0], y = par[1];
			double e = y - fx.avaliar(x);
			s += e * e;
		}
		double mse = s / dadosXY.size();
		return 1.0 / (1.0 + mse);
	}
}

/**
 * Fábrica para criar indivíduos e dados de teste. <lo>
 * <li>Cria indivíduos femininos (XX) e masculinos (XY) com genes aleatórios.</li>
 * <li>Gera dados de teste com base em um polinômio alvo, adicionando ruído gaussiano.</li>
 * <li>Os dados de teste são usados para avaliar o fitness dos indivíduos durante a evolução.</li>
 */
class Fabrica {

	static Feminino novoFeminino() {
		return new Feminino(X.novoAleatorio(), X.novoAleatorio());
	}

	static Masculino novoMasculino() {
		return new Masculino(X.novoAleatorio(), Y.novoAleatorio());
	}

	static List<double[]> dados() {

		List<double[]> ds = new ArrayList<>();
		for (int i = 0; i < Params.NUM_PONTOS; i++) {
			double x = Params.X_MIN + (Params.X_MAX - Params.X_MIN) * i / (Params.NUM_PONTOS - 1.0);
			double y = 0.0, pot = 1.0;
			for (double c : Params.ALVO) {
				y += c * pot;
				pot *= x;
			}
			y += Params.gauss(Params.NOISE_STD);
			ds.add(new double[] { x, y });
		}
		return ds;
	}
}

/**
 * Operadores genéticos para seleção, mutação e reprodução.
 * <ol>
 * <li>Realiza torneios para selecionar indivíduos femininos e masculinos com base no fitness.</li>
 * <li>Aplica mutações nos cromossomos X e Y dos indivíduos, alterando seus genes de forma aleatória.</li>
 * <li>Aplica a tendência de Y no X do filho XY, ajustando os coeficientes de X com base nos deltas de Y.</li>
 * <li>Reproduz indivíduos XX e XY, criando novos indivíduos com base nos pais selecionados.</li>
 * <li>O processo de reprodução envolve a seleção de um X da mãe e um Y do pai, aplicando a tendência de Y no X do filho XY.</li>
 * <li>Os filhos são criados com base na combinação dos genes dos pais, garantindo diversidade genética na população.</li>
 * <li>Os operadores genéticos são fundamentais para a evolução da população, permitindo a exploração de novas soluções e a adaptação ao longo das gerações.</li>
 * </ol>
 */
class Operadores {

	// seleção: torneio dentro do sexo
	static Feminino torneioF(Populacao pop, List<double[]> dados) {
		Random R = Params.RNG;
		Feminino best = null;
		double bf = -1;
		int k = Math.min(Params.TORNEIO_XX, Math.max(1, pop.femininos.size()));
		for (int i = 0; i < k; i++) {
			Feminino cand = pop.femininos.get(R.nextInt(pop.femininos.size()));
			double f = Avaliacao.fitness(cand, dados);
			if (f > bf) {
				bf = f;
				best = cand;
			}
		}
		return (best != null) ? best : Fabrica.novoFeminino();
	}

	static Masculino torneioM(Populacao pop, List<double[]> dados) {
		Random R = Params.RNG;
		Masculino best = null;
		double bf = -1;
		int k = Math.min(Params.TORNEIO_XY, Math.max(1, pop.masculinos.size()));
		for (int i = 0; i < k; i++) {
			Masculino cand = pop.masculinos.get(R.nextInt(pop.masculinos.size()));
			double f = Avaliacao.fitness(cand, dados);
			if (f > bf) {
				bf = f;
				best = cand;
			}
		}
		return (best != null) ? best : Fabrica.novoMasculino();
	}

	// mutações
	static void mutarX(X x) {
		x.mutar();
	}

	static void mutarY(Y y) {
		y.mutar();
	}

	// aplicar tendência de Y no X do filho XY (alinhando pelos comprimentos úteis)
	static void aplicarTendencia(X x, Y y) {

		int L = Math.min(x.lenUtil(), y.lenUtil());
		for (int i = 0; i < L; i++) {
			x.gene[i] = Params.clamp(x.gene[i] + Params.ETA_TREND * y.gene[i],
					Params.COEF_MIN, Params.COEF_MAX);
		}
	}

	// reprodução XX × XY (a cada cópia, telômeros são consumidos ou genes úteis truncados)
	static Individuo reproduzir(Feminino mae, Masculino pai) {

		boolean nasceXY = Params.RNG.nextDouble() < Params.PROB_FILHO_XY;

		// mãe passa um X aleatório (envelhecido)
		X xMae = (Params.RNG.nextBoolean() ? mae.getX1() : mae.getX2()).copiaEnvelhecendo();

		if (nasceXY) {
			// pai passa Y (envelhecido); filho: XY
			Y yPai = pai.getY().copiaEnvelhecendo();
			X xFilho = xMae.copiaSemEnvelhecer(); // x do filho começa como o x da mãe
			aplicarTendencia(xFilho, yPai); // Y guia X
			mutarX(xFilho);
			mutarY(yPai);
			return new Masculino(xFilho, yPai);
		} else {
			// pai passa X (envelhecido); filho: XX
			X xPai = pai.getX().copiaEnvelhecendo();
			mutarX(xMae);
			mutarX(xPai);
			return new Feminino(xMae, xPai);
		}
	}
}


public class EvolutionalInterpolationSimulation {

	/**
	 * Formata os coeficientes de um vetor de coeficientes em uma string legível.
	 * 
	 * @param c
	 * @return
	 */
	static String fmt(double[] c) {
		return IntStream.range(0, c.length)
				.mapToObj(i -> String.format(Locale.US, "a%d=%.3f", i, c[i]))
				.collect(Collectors.joining(", "));
	}

	public static void run() {

		List<double[]> dados = Fabrica.dados();

		// população inicial
		Populacao pop = new Populacao();
		int nXY = (int) Math.round(Params.FRAQ_INICIAL_XY * Params.POP);
		int nXX = Params.POP - nXY;
		for (int i = 0; i < nXX; i++)
			pop.adicionar(Fabrica.novoFeminino());
		for (int i = 0; i < nXY; i++)
			pop.adicionar(Fabrica.novoMasculino());

		// estatística
		Individuo bestEver = null;
		double bestFitEver = -1;

		for (int g = 1; g <= Params.GERACOES; g++) {
			// ranking por sexo
			pop.ranquear(ind -> Avaliacao.fitness(ind, dados));

			Feminino eliteF = pop.rankFeminino.keySet().stream().findFirst().orElse(null);
			Masculino eliteM = pop.rankMasculino.keySet().stream().findFirst().orElse(null);

			double fitF = eliteF != null ? Avaliacao.fitness(eliteF, dados) : -1;
			double fitM = eliteM != null ? Avaliacao.fitness(eliteM, dados) : -1;

			Individuo bestGen = (fitF >= fitM ? eliteF : eliteM);
			double bestFit = Math.max(fitF, fitM);

			if (bestFit > bestFitEver) {
				bestFitEver = bestFit;
				bestEver = bestGen.copiaSemEnvelhecer();
			}

			// logs
			if (g == 1 || g % 10 == 0 || g == Params.GERACOES) {
				long qXX = pop.femininos.size();
				long qXY = pop.masculinos.size();
				System.out.printf(Locale.US,
						"G%3d | best=%.5f | XX=%d XY=%d | coefs(best)=%s%n",
						g, bestFit, qXX, qXY, fmt(bestGen.fenotipoX().gene));
			}

			// --------- nova geração ---------
			List<Feminino> novosF = new ArrayList<>();
			List<Masculino> novosM = new ArrayList<>();

			// elitismo 1 por sexo (elitismo também envelhece na passagem de geração)
			if (eliteF != null)
				novosF.add((Feminino) eliteF.copiaEnvelhecendo());
			if (eliteM != null)
				novosM.add((Masculino) eliteM.copiaEnvelhecendo());

			// garantir reprodutores
			if (pop.femininos.isEmpty())
				pop.adicionar(Fabrica.novoFeminino());
			if (pop.masculinos.isEmpty())
				pop.adicionar(Fabrica.novoMasculino());

			int alvoXY = Math.max(Params.MIN_XY, (int) Math.round(Params.FRAQ_INICIAL_XY * Params.POP));

			while (novosF.size() + novosM.size() < Params.POP) {
				Feminino mae = Operadores.torneioF(pop, dados);
				Masculino pai = Operadores.torneioM(pop, dados);

				Individuo filho = Operadores.reproduzir(mae, pai);
				if (filho instanceof Feminino)
					novosF.add((Feminino) filho);
				else
					novosM.add((Masculino) filho);

				// se XY estiver rareando, viés suave para produzir mais XY
				if (novosM.size() < alvoXY && novosF.size() + novosM.size() < Params.POP) {
					Individuo f2 = Operadores.reproduzir(mae, pai);
					if (f2 instanceof Masculino)
						novosM.add((Masculino) f2);
				}
			}

			// substituição
			pop.limpar();
			// manter proporção, mas prioridade é preencher POP total
			int capXX = Math.min(Params.POP, novosF.size());
			for (int i = 0; i < capXX; i++)
				pop.adicionar(novosF.get(i));

			int capXY = Math.min(Params.POP - pop.femininos.size(), novosM.size());
			for (int i = 0; i < capXY; i++)
				pop.adicionar(novosM.get(i));
		}

		// resultado final (reavalia em dados novos com ruído para sanity check)
		double fitFinal = Avaliacao.fitness(bestEver, Fabrica.dados());
		System.out.println("\n=== Melhor (snapshot) de toda a simulação ===");
		System.out.printf(Locale.US, "fitness=%.6f | coefs=%s%n",
				fitFinal, fmt(bestEver.fenotipoX().gene));
	}
}
