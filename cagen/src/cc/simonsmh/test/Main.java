
/*
 * Copyright 2020 Simon Shi
 *
 * Reference:
 * https://github.com/akiraly/cage/blob/master/cage-examples-parent/cage-e01-simple/src/main/java/com/github/cage/examples/cage_e01_simple/App.java
 *
 */
package cc.simonsmh.test;

import com.github.cage.Cage;
import com.github.cage.GCage;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An edited example console application that generates a few hundreds captcha images
 * and writes them to folder.
 *
 * @author simonsmh
 */
public class Main {
    public static void main(String[] args) throws IOException {

        for (int i = 0; i < 100; i++) {
            generate(new GCage(), 400, "data/train", String.format("%02d", i) + "_", ".jpg", String.format("%02d", i));
        }
        for (int i = 0; i < 100; i++) {
            generate(new GCage(), 100, "data/valid", String.format("%02d", i) + "_", ".jpg", String.format("%02d", i));
        }
        for (int i = 0; i < 100; i++) {
            generate(new GCage(), 100, "data/test", String.format("%02d", i) + "_", ".jpg", String.format("%02d", i));
        }
    }

    protected static void generate(Cage cage, int num, String nameDir, String namePrefix,
                                   String namePostfix, String text) throws IOException {
        for (int fi = 0; fi < num; fi++) {
            File targetDir = new File(nameDir);
            File targetFile = new File(targetDir, namePrefix + fi + namePostfix);
            final OutputStream os = new FileOutputStream(targetFile, false);
            try {
                cage.draw(
                        text != null ? text : cage.getTokenGenerator().next(),
                        os);
            } finally {
                os.close();
            }
        }
    }
}

