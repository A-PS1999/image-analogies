#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <vector>
#include <memory>
#include <map>

namespace Util
{
    namespace ArgParser
    {
        struct Argument
        {
            std::string name;
            std::string description;
            std::string value;
            bool required = false;
        };

        class ArgumentParser
        {
        public:
            void addArgument(const std::string &name, const std::string &description, bool required = false, std::string defaultValue = "");
            bool parseInput(int argc, char **argv);
            const Argument *getArgument(const std::string &name) const;
            void printUsage() const;

        private:
            std::vector<std::unique_ptr<Argument>> arguments;
            std::map<std::string, Argument *> argMap;
            int parsedArgsCount = 0;
            int requiredArgsCount = 0;
            std::string programName;
        };

        void addBasicArguments(ArgumentParser &parser);
        float parseCoherenceWeight(const ArgumentParser &parser);
    } // namespace ArgParser
} // namespace Util

#endif // ARG_PARSER_H