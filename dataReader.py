import os


class dataReader:

    def readFiles(self, path, fileType):
        # reading input files
        Files = []
        for r, d, f in os.walk(path):
            for file in f:
                if fileType in file:
                    Files.append(os.path.join(r, file))

        Files.sort()

        return Files


