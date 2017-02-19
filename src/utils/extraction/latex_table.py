import cPickle as pickle


def generate_table(filename, out_file):

    res_dict = pickle.load(open(filename, "rb"))

    time_unit = 'min' if res_dict['dataset'] == 'mnist' else 'h'
    time_norm = 60. if time_unit == 'min' else 3600.

    latex_code = """\\documentclass{{article}}
    \\usepackage[utf8]{{inputenc}}

    \\title{{Impressive Results}}

    \\begin{{document}}

    \\maketitle

    \\begin{{table}}[h]
    \\begin{{center}}
    \\begin{{tabular}}{{lcccr}}
    {{\\bf Solver (epochs)}}  &{{\\bf Training}} &{{\\bf Training }}
    &{{\\bf Time ({})}} &{{\\bf Testing}} \\\ \
    &{{\\bf Objective}} &{{\\bf Accuracy}} & &{{\\bf Accuracy}}
    \\\ \\hline
    """.format(time_unit)

    for solver in ["adagrad", "adadelta", "adam"]:
        lwsvm_solver = "{}_lwsvm".format(solver)
        full_solver = "{}_full".format(solver)

        newline = "%s (%i) &%.3f &%.2f\\%% &%.2f &%.2f\\%% \\\ \n" \
            % (solver.title(),
               res_dict[solver]["n_epochs"],
               res_dict[solver]["train_objective"],
               res_dict[solver]["train_accuracy"],
               res_dict[solver]["time"] / time_norm,
               res_dict[solver]["test_accuracy"])

        latex_code += newline

        newline = "%s (%i) &%.3f &%.2f\\%% &%.2f &%.2f\\%% \\\ \n" \
            % (solver.title(),
               res_dict[full_solver]["n_epochs"],
               res_dict[full_solver]["train_objective"],
               res_dict[full_solver]["train_accuracy"],
               res_dict[full_solver]["time"] / time_norm,
               res_dict[full_solver]["test_accuracy"])

        latex_code += newline

        newline = "%s (%i) + LW-SVM &%.3f &%.2f\\%% &%.2f+%.2f &%.2f\\%% \\\ \\hline \n" \
            % (solver.title(),
               res_dict[solver]["n_epochs"],
               res_dict[lwsvm_solver]["train_objective"],
               res_dict[lwsvm_solver]["train_accuracy"],
               res_dict[solver]["time"] / time_norm,
               res_dict[lwsvm_solver]["time"] / time_norm,
               res_dict[lwsvm_solver]["test_accuracy"])
        latex_code += newline

    latex_code += """\\end{tabular}
    \\end{center}
    \\end{table}

    \\end{document}
    """

    with open(out_file, "w") as text_file:
        text_file.write(latex_code)

    print("Results written as latex code in {}".format(out_file))
