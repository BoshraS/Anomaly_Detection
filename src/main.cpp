#include "conf/config.h"
#include "cmd/cmd.h"

using namespace NIBR;

int main(int argc, char *argv[]) {

    NIBR::INITIALIZE();

    // Parse input
    CLI::App app("Anomap v0.1");
    app.footer("----------Anomap v0.1---------\n© Copyright 2024, Anomap Team\n");
    // app.failure_message(CLI::FailureMessage::help);

    app.require_subcommand(1);

    disp(MSG_DEBUG, "Adding Anomap commands");

    encode(app.add_subcommand("encode", ""));
    decode(app.add_subcommand("decode", ""));
    findClusterCenters(app.add_subcommand("findClusterCenters", ""));
    findClusterCenters_euclidean(app.add_subcommand("findClusterCentersEuclidean", ""));
    findClusterCenters_latentBrute(app.add_subcommand("findClusterCentersLatentBrute", ""));
    score(app.add_subcommand("score", ""));
    toTrack(app.add_subcommand("toTrack", ""));
    toImg(app.add_subcommand("toImg", ""));
    modelTest(app.add_subcommand("modelTest", ""));
    modelTest_precalc(app.add_subcommand("modelTest_precalc", ""));
    measureDistance(app.add_subcommand("measureDistance", ""));
    compareSpeed(app.add_subcommand("compareSpeed", ""));
    compareDistance(app.add_subcommand("compareDistance", ""));

    disp(MSG_DEBUG, "Parsing input");

    // If no option is used just display the help
    try {

        app.parse(argc, argv);

    } catch(const CLI::ParseError &e) {

        // Check if a subcommand is run with no arguments/options, then display help
        CLI::App* subcmd = &app;
        int argCount = 1;

        while (subcmd->get_subcommands().size()>0) {
            subcmd   = subcmd->get_subcommands()[0];
            argCount = subcmd->count_all();
        }

        if (argCount==1) { // Check if subcmd is run with no arguments/options
            displayHelp(app.help());
            NIBR::TERMINATE();
            return EXIT_SUCCESS;
        }

        auto q = app.exit(e);

        NIBR::TERMINATE();
        return q;

    }

    NIBR::TERMINATE();
    
    return EXIT_SUCCESS;

}

