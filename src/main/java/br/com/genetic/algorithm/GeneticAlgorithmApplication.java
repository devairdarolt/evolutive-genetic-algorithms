package br.com.genetic.algorithm;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import br.com.genetic.algorithm.interpolation.EvolutionalInterpolationSimulation;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@SpringBootApplication
public class GeneticAlgorithmApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(GeneticAlgorithmApplication.class, args);
	}

	@Override
	public void run(String... args) throws Exception {

		log.info("Iniciando o Algoritmo Genético...");
		double[] coef = new double[] { 1.0, 2.0, 0.3 }; // Coeficientes do polinômio alvo: 1.0 + 2.0*x + 0.3*x^2
		EvolutionalInterpolationSimulation.run(coef);
		log.info("Algoritmo Genético finalizado.");

	}

}
