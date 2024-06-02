const { defineConfig } = require("@vue/cli-service");
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    hot: true, // Ensure hot module replacement is enabled
    open: true, // Automatically open the browser
  },
});
