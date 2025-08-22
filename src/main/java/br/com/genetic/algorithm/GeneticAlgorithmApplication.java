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
		EvolutionalInterpolationSimulation.run();

	}

}
