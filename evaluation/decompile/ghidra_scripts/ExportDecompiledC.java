// Export the decompiled C of every function to one file.
//
// A Ghidra post-script run under analyzeHeadless. It decompiles every function
// with the Decompiler and concatenates each function's C into a single
// translation unit, written to the path given as the first script arg. Written
// in Java (not Python) so it runs headless without PyGhidra, which Ghidra 12
// requires for Python scripts.
//
//   analyzeHeadless <proj> <name> -import <bin> \
//       -scriptPath <this dir> -postScript ExportDecompiledC.java <out.c>
//
//@category DEFLATE-D
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.decompiler.DecompiledFunction;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.util.task.ConsoleTaskMonitor;

public class ExportDecompiledC extends GhidraScript {

    private static final int DECOMPILE_TIMEOUT = 120; // seconds per function

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 1) {
            println("ExportDecompiledC: missing output path argument");
            return;
        }
        String outPath = args[0];

        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);
        ConsoleTaskMonitor monitor = new ConsoleTaskMonitor();

        FunctionManager fm = currentProgram.getFunctionManager();
        List<Function> funcs = new ArrayList<>();
        for (Function f : fm.getFunctions(true)) {
            funcs.add(f);
        }
        // Stable order by entry address so runs are reproducible.
        funcs.sort(Comparator.comparingLong(
                f -> f.getEntryPoint().getOffset()));

        StringBuilder sb = new StringBuilder();
        int ok = 0;
        for (Function f : funcs) {
            DecompileResults res = decomp.decompileFunction(f, DECOMPILE_TIMEOUT, monitor);
            if (res != null && res.decompileCompleted()) {
                DecompiledFunction df = res.getDecompiledFunction();
                if (df != null) {
                    String c = df.getC();
                    if (c != null && !c.isEmpty()) {
                        sb.append(c).append("\n");
                        ok++;
                    }
                }
            }
        }
        decomp.dispose();

        try (Writer w = new OutputStreamWriter(
                new FileOutputStream(outPath), StandardCharsets.UTF_8)) {
            w.write(sb.toString());
        }
        println("ExportDecompiledC: " + ok + "/" + funcs.size()
                + " functions -> " + outPath + " (" + sb.length() + " chars)");
    }
}
