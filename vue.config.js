const path = require('path');

module.exports = {
  devServer: {
    proxy: {
      '/data_upload': {
        target: 'http://localhost:8080',
        pathRewrite: { '^/data_upload': '' }
      }
    }
  },
  configureWebpack: {
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src')
      }
    }
  }
};
