package com.example.springboot_sign;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Size;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.DFT_INVERSE;
import static org.bytedeco.javacpp.opencv_core.DFT_REAL_OUTPUT;
import static org.bytedeco.javacpp.opencv_core.DFT_SCALE;
import static org.bytedeco.javacpp.opencv_core.NORM_MINMAX;
import static org.bytedeco.javacpp.opencv_core.Scalar;
import static org.bytedeco.javacpp.opencv_core.add;
import static org.bytedeco.javacpp.opencv_core.addWeighted;
import static org.bytedeco.javacpp.opencv_core.copyMakeBorder;
import static org.bytedeco.javacpp.opencv_core.dft;
import static org.bytedeco.javacpp.opencv_core.flip;
import static org.bytedeco.javacpp.opencv_core.getOptimalDFTSize;
import static org.bytedeco.javacpp.opencv_core.idft;
import static org.bytedeco.javacpp.opencv_core.log;
import static org.bytedeco.javacpp.opencv_core.magnitude;
import static org.bytedeco.javacpp.opencv_core.merge;
import static org.bytedeco.javacpp.opencv_core.normalize;
import static org.bytedeco.javacpp.opencv_core.split;
import static org.bytedeco.javacpp.opencv_core.subtract;
import static org.bytedeco.javacpp.opencv_core.multiply;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FONT_HERSHEY_COMPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

@SpringBootApplication
public class BlindWaterMark {

    private static final double DEFAULT_STRENGTH = 0.18;
    private static final double[] DEFAULT_BAND_SET = new double[] { 0.28, 0.38, 0.52 };
    private static final double[] DEFAULT_STRENGTH_DISTRIBUTION = new double[] { 0.35, 0.45, 0.20 };

    public static void main(String[] args) {
        Cli cli = Cli.parse(args);
        BlindWaterMark bwm = new BlindWaterMark();
        switch (cli.type) {
            case ENCODE:
                bwm.encode(cli.encodeConfig);
                System.out.println("ENCODE SUCCESSFUL");
                break;
            case DECODE:
                bwm.decode(cli.decodeConfig);
                System.out.println("DECODE SUCCESSFUL");
                break;
            default:
                help();
        }
    }

    private void encode(EncodeConfig config) {
        Mat src = loadColor(config.sourceTexture);
        MatVector colorPlanes = new MatVector();
        split(src, colorPlanes);

        Mat geometryWeights = null;
        Mat imagePayload = null;
        if (config.mode == Mode.IMAGE) {
            imagePayload = prepareImagePayload(config.payload);
        }

        for (int i = 0; i < colorPlanes.size(); i++) {
            Mat channel = colorPlanes.get(i);
            Size originalSize = channel.size();

            Mat optimized = optimizedImage(toFloat(channel));
            Mat spectrum = startDFT(optimized);

            if (geometryWeights == null) {
                geometryWeights = config.geometryMap == null
                        ? Mat.ones(spectrum.size(), CV_32F).asMat()
                        : prepareGeometryWeights(spectrum.size(), config.geometryMap);
            }

            embedPayloadMultiBand(spectrum, config, geometryWeights, imagePayload, i);

            Mat restored = inverseDFT(spectrum, originalSize);
            colorPlanes.put(i, restored);
        }

        Mat encoded = new Mat();
        merge(colorPlanes, encoded);
        imwrite(config.output, encoded);
    }

    private void decode(DecodeConfig config) {
        if (config.mode == Mode.TEXT) {
            decodeText(config.watermarkedTexture, config.output);
        } else {
            decodeImage(config.sourceTexture, config.watermarkedTexture, config.output);
        }
    }

    private void decodeText(String encodedTexture, String output) {
        Mat encoded = toFloat(loadGray(encodedTexture));
        Mat optimized = optimizedImage(encoded);
        Mat spectrum = startDFT(optimized);
        Mat magnitude = magnitudeFromComplex(spectrum);
        shiftDFT(magnitude);
        normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1, null);
        magnitude.convertTo(magnitude, CV_8UC1);
        imwrite(output, magnitude);
    }

    private void decodeImage(String sourceTexture, String encodedTexture, String output) {
        Mat original = toFloat(loadGray(sourceTexture));
        Mat encoded = toFloat(loadGray(encodedTexture));

        Mat originalSpectrum = startDFT(optimizedImage(original));
        Mat encodedSpectrum = startDFT(optimizedImage(encoded));

        Mat diff = new Mat();
        subtract(encodedSpectrum, originalSpectrum, diff);
        Mat recovered = inverseDFT(diff, original.size());
        imwrite(output, recovered);
    }

    private static void embedPayloadMultiBand(Mat spectrum,
            EncodeConfig config,
            Mat geometryWeights,
            Mat imagePayload,
            int channelIndex) {
        for (int band = 0; band < config.bandRatios.length; band++) {
            double ratio = config.bandRatios[band];
            double strength = config.strengths[band];
            long seed = seedForBand(config, channelIndex, band);
            if (config.mode == Mode.TEXT) {
                embedTextPayload(spectrum, config.payload, strength, ratio, geometryWeights, seed);
            } else {
                embedImagePayload(spectrum, imagePayload, config.payload, strength, ratio, geometryWeights, seed);
            }
        }
    }

    private static long seedForBand(EncodeConfig config, int channelIndex, int bandIndex) {
        return Objects.hash(config.payload, config.key, channelIndex, bandIndex);
    }

    private static Mat startDFT(Mat srcImg) {
        MatVector planes = new MatVector(2);
        Mat comImg = new Mat();
        planes.put(0, srcImg);
        planes.put(1, Mat.zeros(srcImg.size(), CV_32F).asMat());
        merge(planes, comImg);
        dft(comImg, comImg);
        return comImg;
    }

    private static Mat inverseDFT(Mat spectrum, Size originalSize) {
        Mat spatial = new Mat();
        idft(spectrum, spatial, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE, 0);
        Rect roi = new Rect(0, 0, originalSize.width(), originalSize.height());
        Mat cropped;
        try (Mat roiView = new Mat(spatial, roi)) {
            cropped = roiView.clone();
        }
        normalize(cropped, cropped, 0.0, 255.0, NORM_MINMAX, -1, null);
        cropped.convertTo(cropped, CV_8UC1);
        return cropped;
    }

    private static Mat optimizedImage(Mat srcImg) {
        Mat padded = new Mat();
        int opRows = getOptimalDFTSize(srcImg.rows());
        int opCols = getOptimalDFTSize(srcImg.cols());
        copyMakeBorder(srcImg, padded, 0, opRows - srcImg.rows(),
                0, opCols - srcImg.cols(), BORDER_CONSTANT, Scalar.all(0));
        return padded;
    }

    private static void shiftDFT(Mat comImg) {
        comImg = new Mat(comImg, new Rect(0, 0, comImg.cols() & -2, comImg.rows() & -2));
        int cx = comImg.cols() / 2;
        int cy = comImg.rows() / 2;

        Mat q0 = new Mat(comImg, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(comImg, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(comImg, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(comImg, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    private static Mat prepareImagePayload(String watermarkPath) {
        Mat wm = toFloat(loadGray(watermarkPath));
        normalize(wm, wm, 0.0, 1.0, NORM_MINMAX, -1, null);
        return wm;
    }

    private static Mat prepareGeometryWeights(Size size, String geometryPath) {
        Mat weights = toFloat(loadGray(geometryPath));
        normalize(weights, weights, 0.0, 1.0, NORM_MINMAX, -1, null);
        Mat resized = new Mat();
        resize(weights, resized, size);
        return resized;
    }

    private static void embedTextPayload(Mat spectrum, String payload, double strength, double bandRatio,
            Mat geometryWeights, long seed) {
        shiftDFT(spectrum);
        Mat mask = Mat.zeros(spectrum.size(), CV_32F).asMat();
        Random random = new Random(seed);
        Rect bandRect = createBandRect(mask.size(), bandRatio, random);
        Mat roi = new Mat(mask, bandRect);
        double fontScale = clamp(0.6 + bandRatio, 0.7, 2.2);
        int thickness = Math.max(1, (int) Math.round(2 + bandRatio * 2));
        Point anchor = new Point(
                Math.max(1, bandRect.width() / 8 + random.nextInt(Math.max(1, bandRect.width() / 5))),
                Math.max(thickness + 1, bandRect.height() / 2 + random.nextInt(Math.max(1, bandRect.height() / 5))
                        - bandRect.height() / 10));
        putText(roi, payload, anchor, CV_FONT_HERSHEY_COMPLEX, fontScale, Scalar.all(1.0), thickness, CV_AA, false);
        Mat flipped = new Mat();
        flip(roi, flipped, -1);
        addWeighted(roi, 0.5, flipped, 0.5, 0.0, roi);
        mirrorSpectrumMask(mask);
        applyGeometryMask(mask, geometryWeights);
        Mat complexMask = asComplex(mask);
        addWeighted(complexMask, strength, spectrum, 1.0, 0.0, spectrum);
        shiftDFT(spectrum);
    }

    private static void embedImagePayload(Mat spectrum, Mat watermarkPayload, String watermarkPath,
            double strength, double bandRatio, Mat geometryWeights, long seed) {
        if (watermarkPayload == null) {
            throw new IllegalStateException("Watermark payload not preloaded for " + watermarkPath);
        }
        shiftDFT(spectrum);
        Mat watermark = watermarkPayload.clone();
        Random random = new Random(seed);
        Rect bandRect = createBandRect(spectrum.size(), bandRatio, random);
        Mat mask = Mat.zeros(spectrum.size(), CV_32F).asMat();
        Mat roi = new Mat(mask, bandRect);
        Mat resized = new Mat();
        resize(watermark, resized, new Size(bandRect.width(), bandRect.height()));
        resized.copyTo(roi);
        mirrorSpectrumMask(mask);
        applyGeometryMask(mask, geometryWeights);
        Mat complexMask = asComplex(mask);
        addWeighted(complexMask, strength, spectrum, 1.0, 0.0, spectrum);
        shiftDFT(spectrum);
    }

    private static Mat magnitudeFromComplex(Mat complex) {
        MatVector planes = new MatVector(2);
        Mat mag = new Mat();
        split(complex, planes);
        magnitude(planes.get(0), planes.get(1), mag);
        add(Mat.ones(mag.size(), CV_32F).asMat(), mag, mag);
        log(mag, mag);
        return mag;
    }

    private static Rect createBandRect(Size size, double ratio, Random random) {
        int width = Math.max(4, (int) Math.round(size.width() * ratio));
        int height = Math.max(4, (int) Math.round(size.height() * ratio));
        int baseX = Math.max(0, (size.width() - width) / 2);
        int baseY = Math.max(0, (size.height() - height) / 2);
        int jitterX = Math.max(1, baseX / 2 + 1);
        int jitterY = Math.max(1, baseY / 2 + 1);
        int x = clampInt(baseX + random.nextInt(jitterX * 2) - jitterX, 0, size.width() - width);
        int y = clampInt(baseY + random.nextInt(jitterY * 2) - jitterY, 0, size.height() - height);
        return new Rect(x, y, width, height);
    }

    private static void applyGeometryMask(Mat mask, Mat geometryWeights) {
        if (geometryWeights == null) {
            return;
        }
        multiply(mask, geometryWeights, mask);
    }

    private static int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void mirrorSpectrumMask(Mat mask) {
        Mat flipped = new Mat();
        flip(mask, flipped, -1);
        addWeighted(mask, 0.5, flipped, 0.5, 0.0, mask);
    }

    private static Mat asComplex(Mat magnitudePlane) {
        MatVector planes = new MatVector(2);
        planes.put(0, magnitudePlane);
        planes.put(1, Mat.zeros(magnitudePlane.size(), CV_32F).asMat());
        Mat complex = new Mat();
        merge(planes, complex);
        return complex;
    }

    private static Mat loadColor(String path) {
        Mat img = imread(path, CV_LOAD_IMAGE_COLOR);
        ensureNotEmpty(img, path);
        return img;
    }

    private static Mat loadGray(String path) {
        Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        ensureNotEmpty(img, path);
        return img;
    }

    private static void ensureNotEmpty(Mat img, String path) {
        if (img == null || img.empty()) {
            throw new IllegalArgumentException("Unable to read image: " + path);
        }
    }

    private static Mat toFloat(Mat src) {
        Mat dst = new Mat();
        src.convertTo(dst, CV_32F);
        return dst;
    }

    private static EncodeConfig buildEncodeConfig(Map<String, String> options) {
        Mode mode = Mode.from(options.get("--mode"));
        String src = options.get("--src");
        String output = options.get("--out");
        String payload = options.get("--wm");
        if (src == null || output == null || payload == null) {
            help();
        }
        double baseStrength = clamp(parseDouble(options.get("--strength"), DEFAULT_STRENGTH), 0.05, 0.6);
        double[] bandRatios;
        if (options.containsKey("--band")) {
            bandRatios = new double[] { clamp(parseDouble(options.get("--band"), DEFAULT_BAND_SET[1]), 0.1, 0.85) };
        } else {
            bandRatios = clampArray(parseDoubleArray(options.get("--bands"), DEFAULT_BAND_SET), 0.1, 0.85);
        }
        double[] strengthShape = alignShape(parseDoubleArray(options.get("--strengths"), DEFAULT_STRENGTH_DISTRIBUTION),
                bandRatios.length);
        double[] strengths = scaleStrengths(baseStrength, strengthShape);
        String geometry = options.get("--geometry");
        String key = options.getOrDefault("--key", payload);
        return new EncodeConfig(mode, src, payload, output, bandRatios, strengths, geometry, key);
    }

    private static DecodeConfig buildDecodeConfig(Map<String, String> options) {
        Mode mode = Mode.from(options.get("--mode"));
        String output = options.get("--out");
        String watermarked = options.get("--wm");
        if (output == null || watermarked == null) {
            help();
        }
        if (mode == Mode.IMAGE) {
            String source = options.get("--src");
            if (source == null) {
                help();
            }
            return new DecodeConfig(mode, source, watermarked, output);
        }
        return new DecodeConfig(mode, null, watermarked, output);
    }

    private static double parseDouble(String raw, double defaultValue) {
        if (raw == null) {
            return defaultValue;
        }
        return Double.parseDouble(raw);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private static double[] parseDoubleArray(String raw, double[] defaults) {
        if (raw == null) {
            return defaults.clone();
        }
        String[] tokens = raw.split(",");
        double[] values = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            values[i] = Double.parseDouble(tokens[i].trim());
        }
        return values;
    }

    private static double[] clampArray(double[] values, double min, double max) {
        double[] clamped = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            clamped[i] = clamp(values[i], min, max);
        }
        return clamped;
    }

    private static double[] alignShape(double[] shape, int targetLength) {
        if (shape.length == targetLength) {
            return shape;
        }
        if (shape.length == 1) {
            double[] expanded = new double[targetLength];
            Arrays.fill(expanded, shape[0]);
            return expanded;
        }
        if (targetLength == 1) {
            return new double[] { Arrays.stream(shape).average().orElse(shape[0]) };
        }
        help();
        return shape;
    }

    private static double[] scaleStrengths(double totalStrength, double[] shape) {
        double sum = Arrays.stream(shape).sum();
        if (sum == 0) {
            double equal = totalStrength / shape.length;
            double[] uniform = new double[shape.length];
            Arrays.fill(uniform, equal);
            return uniform;
        }
        double[] result = new double[shape.length];
        for (int i = 0; i < shape.length; i++) {
            result[i] = totalStrength * (shape[i] / sum);
        }
        return result;
    }

    private static void help() {
        System.out.println("Usage:\n" +
                "  encode --mode <text|image> --src <texture.png> --wm <text | wm.png> --out <output.png>\n" +
                "         [--strength 0.18] [--bands 0.28,0.38,0.52 | --band 0.35]\n" +
                "         [--strengths 0.35,0.45,0.2] [--geometry geom_map.png] [--key secret]\n" +
                "  decode --mode text  --wm <encoded_texture.png>               --out <spectrum.png>\n" +
                "  decode --mode image --src <original_texture.png> --wm <encoded_texture.png> --out <recovered.png>\n");
        System.exit(-1);
    }

    private enum Mode {
        TEXT, IMAGE;

        static Mode from(String raw) {
            if (raw == null) {
                help();
            }
            switch (raw.toLowerCase(Locale.ROOT)) {
                case "text":
                    return TEXT;
                case "image":
                    return IMAGE;
                default:
                    help();
                    return TEXT;
            }
        }
    }

    private enum CommandType {
        ENCODE, DECODE
    }

    private static final class EncodeConfig {
        private final Mode mode;
        private final String sourceTexture;
        private final String payload;
        private final String output;
        private final double[] bandRatios;
        private final double[] strengths;
        private final String geometryMap;
        private final String key;

        private EncodeConfig(Mode mode, String sourceTexture, String payload, String output,
                double[] bandRatios, double[] strengths, String geometryMap, String key) {
            this.mode = mode;
            this.sourceTexture = sourceTexture;
            this.payload = payload;
            this.output = output;
            this.bandRatios = Arrays.copyOf(bandRatios, bandRatios.length);
            this.strengths = Arrays.copyOf(strengths, strengths.length);
            this.geometryMap = geometryMap;
            this.key = key;
        }
    }

    private static final class DecodeConfig {
        private final Mode mode;
        private final String sourceTexture;
        private final String watermarkedTexture;
        private final String output;

        private DecodeConfig(Mode mode, String sourceTexture, String watermarkedTexture, String output) {
            this.mode = mode;
            this.sourceTexture = sourceTexture;
            this.watermarkedTexture = watermarkedTexture;
            this.output = output;
        }
    }

    private static final class Cli {
        private final CommandType type;
        private final EncodeConfig encodeConfig;
        private final DecodeConfig decodeConfig;

        private Cli(CommandType type, EncodeConfig encodeConfig, DecodeConfig decodeConfig) {
            this.type = type;
            this.encodeConfig = encodeConfig;
            this.decodeConfig = decodeConfig;
        }

        private static Cli parse(String[] args) {
            if (args == null || args.length == 0) {
                help();
            }
            CommandType type = parseCommand(args[0]);
            Map<String, String> options = parseOptions(Arrays.copyOfRange(args, 1, args.length));
            if (type == CommandType.ENCODE) {
                return new Cli(type, buildEncodeConfig(options), null);
            }
            return new Cli(type, null, buildDecodeConfig(options));
        }

        private static CommandType parseCommand(String raw) {
            if ("encode".equalsIgnoreCase(raw)) {
                return CommandType.ENCODE;
            }
            if ("decode".equalsIgnoreCase(raw)) {
                return CommandType.DECODE;
            }
            help();
            return CommandType.ENCODE;
        }

        private static Map<String, String> parseOptions(String[] args) {
            Map<String, String> options = new HashMap<>();
            for (int i = 0; i < args.length; i++) {
                String key = args[i];
                if (!key.startsWith("--")) {
                    help();
                }
                if (i + 1 >= args.length) {
                    help();
                }
                options.put(key, args[++i]);
            }
            return options;
        }
    }
}
