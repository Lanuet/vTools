package edu.ktlab.nlp.chunk;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;

import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkSampleStream;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.chunker.DefaultChunkerContextGenerator;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

public class vChunkerME {
	static int cutoff = 1;
	static int iteration = 200;

	private static ObjectStream<ChunkSample> createSampleStream() throws IOException {
		Charset charset = Charset.forName("UTF-8");
		ObjectStream<String> lineStream = new PlainTextByLineStream(new FileInputStream(
				"data/Chunk/vi-chunk.train"), charset);
		return new ChunkSampleStream(lineStream);
	}

	static ChunkerModel trainChunkModel() throws Exception {
		return ChunkerME.train("vi", createSampleStream(), cutoff, iteration,
				new DefaultChunkerContextGenerator());
	}
    public static void match(String glob, String location) throws IOException {
        final PathMatcher pathMatcher = FileSystems.getDefault().getPathMatcher(glob);
        Files.walkFileTree(Paths.get(location), new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path path, BasicFileAttributes attrs) throws IOException {
                if (pathMatcher.matches(path)) {
                    System.out.println(path);

                    process(path.toString());
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
                return FileVisitResult.CONTINUE;
            }
        });
    }
    static void process(String inputFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(inputFile));
        InputStream inChunk = new FileInputStream("models/Chunk/vi-chunk.model");
        InputStream inPos = new FileInputStream("models/POS/vi-pos.model");
        ChunkerModel chunkModel = new ChunkerModel(inChunk);
        //chunkModel = new ChunkerModel(inChunk);
        POSModel posModel = new POSModel(inPos);

        ChunkerME chunker = new ChunkerME(chunkModel);
        POSTagger tagger = new POSTaggerME(posModel);
        try {
            FileOutputStream fos = new FileOutputStream(inputFile.replace(".txt", "_tokenized.txt"));
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fos, "utf-8"));
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();
            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
            }
            String text = sb.toString();
            String[] tokens = WhitespaceTokenizer.INSTANCE.tokenize(text);
            String[] postags = tagger.tag(tokens);
            String[] chunktags = chunker.chunk(tokens, postags);

            for (int i = 0; i < tokens.length; i++){
                bw.write(tokens[i] + "/" + postags[i] + "/" + chunktags[i]);
                bw.newLine();
            }

            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        inChunk.close();
        inPos.close();
    }

	public static void main(String[] args) throws Exception {
		/*ChunkerModel chunkModel = trainChunkModel();
		OutputStream modelOut = new BufferedOutputStream(new FileOutputStream(
				"models/Chunk/vi-chunker.model"));
		chunkModel.serialize(modelOut);
		modelOut.close();*/


        String glob = "glob:**/*.txt";
        String path = "D:\\Khoa luan\\Models\\Data\\Train";
        match(glob, path);
	}

}
