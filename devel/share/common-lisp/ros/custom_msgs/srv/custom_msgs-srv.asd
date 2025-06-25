
(cl:in-package :asdf)

(defsystem "custom_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ChatPrompt" :depends-on ("_package_ChatPrompt"))
    (:file "_package_ChatPrompt" :depends-on ("_package"))
  ))